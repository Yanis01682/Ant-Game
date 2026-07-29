from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig, infer_observation_dim
from SDK.backend.model import Operation
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import BASE_UPGRADE_COST, MAX_ACTIONS, AntKind, OperationType, PLAYER_BASES, SUPER_WEAPON_STATS, SuperWeaponType, TowerType
from SDK.utils.geometry import hex_distance
from SDK.utils.turns import DecisionContext


def _patch_backend_state() -> None:
    if getattr(BackendState, "_zz_agent_patch_applied", False):
        return

    original_can_apply = BackendState.can_apply_operation

    def patched_can_apply(self, player: int, operation, pending=None) -> bool:
        op_type = getattr(operation, "op_type", getattr(operation, "type", None))
        if op_type == OperationType.UPGRADE_GENERATION_SPEED and self.bases[player].generation_level >= 2:
            return False
        if op_type == OperationType.UPGRADE_GENERATED_ANT and self.bases[player].ant_level >= 2:
            return False
        try:
            return original_can_apply(self, player, operation, pending)
        except IndexError:
            return False

    BackendState.can_apply_operation = patched_can_apply
    BackendState._zz_agent_patch_applied = True


@dataclass(slots=True)
class SearchProfile:
    iterations: int
    max_depth: int
    root_action_limit: int
    child_action_limit: int
    c_puct: float
    prior_mix: float
    value_mix: float


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        iterations: int = 64,
        max_depth: int = 4,
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        model_path: str | os.PathLike[str] | None = None,
        shortlist_size: int = 14,
    ) -> None:
        _patch_backend_state()
        super().__init__(seed=seed, max_actions=max_actions)
        self.base_iterations = iterations
        self.base_depth = max_depth
        self.shortlist_size = max(8, shortlist_size)
        self.model = self._load_model(model_path)
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=SearchConfig(
                iterations=iterations,
                max_depth=max_depth,
                c_puct=1.2,
                root_action_limit=min(self.shortlist_size, 16),
                child_action_limit=8,
                prior_mix=0.75,
                value_mix=0.7,
                seed=seed or 0,
            ),
            feature_extractor=self.feature_extractor,
            action_catalog=self.catalog,
        )

    def _candidate_model_paths(self, override: str | os.PathLike[str] | None) -> list[Path]:
        candidates: list[Path] = []
        if override is not None:
            return [Path(override)]
        env_path = os.getenv("AGENT_TRADITION_MCTS_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent
        candidates.extend(
            [
                module_root / "ai_mcts_model.npz",
                repo_root / "checkpoints" / "ai_mcts_latest.npz",
                repo_root / "SDK" / "checkpoints" / "ai_mcts_latest.npz",
            ]
        )
        return candidates

    def _load_model(self, model_path: str | os.PathLike[str] | None) -> PolicyValueNet | None:
        expected_obs_dim = infer_observation_dim(self.feature_extractor, self.catalog.max_actions)
        seen_existing = False
        for candidate in self._candidate_model_paths(model_path):
            if not candidate.exists():
                continue
            seen_existing = True
            try:
                model = PolicyValueNet.from_checkpoint(candidate)
            except (OSError, ValueError, KeyError) as exc:
                print(f"[mcts] failed to load model {candidate}: {exc}", file=sys.stderr)
                continue
            if model.action_dim != self.catalog.max_actions:
                print(f"[mcts] skipped model {candidate}: action_dim mismatch", file=sys.stderr)
                continue
            if model.obs_dim != expected_obs_dim and not model.resize_observation_dim(expected_obs_dim):
                print(f"[mcts] skipped model {candidate}: observation_dim mismatch", file=sys.stderr)
                continue
            return model
        if model_path is not None or seen_existing:
            print("[mcts] using heuristic policy without neural checkpoint", file=sys.stderr)
        return None

    def _search_profile(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> SearchProfile:
        round_index = state.round_index
        bundle_count = len(bundles)
        timeout_edge = self._timeout_edge(state, player)
        has_weapon = any("weapon" in bundle.tags for bundle in bundles)
        near_storm = 72 <= state.coins[player] < 90 and state.weapon_cooldowns[player][1] == 0
        tactical_turn = has_weapon or near_storm or abs(timeout_edge) >= 90.0
        if round_index < 80:
            return SearchProfile(
                iterations=max(self.base_iterations + 12, 32) if tactical_turn else max(self.base_iterations - 12, 12),
                max_depth=max(self.base_depth - 1, 3),
                root_action_limit=min(bundle_count, 10 if tactical_turn else 8),
                child_action_limit=5 if tactical_turn else 4,
                c_puct=1.05,
                prior_mix=0.82,
                value_mix=0.60,
            )
        if round_index < 220:
            return SearchProfile(
                iterations=min(self.base_iterations + 16, 48) if tactical_turn else max(self.base_iterations - 8, 16),
                max_depth=self.base_depth,
                root_action_limit=min(bundle_count, 12 if tactical_turn else 9),
                child_action_limit=6 if tactical_turn else 4,
                c_puct=1.15,
                prior_mix=0.76,
                value_mix=0.70,
            )
        if timeout_edge < 0.0:
            return SearchProfile(
                iterations=min(self.base_iterations, 56),
                max_depth=self.base_depth,
                root_action_limit=min(bundle_count, 12),
                child_action_limit=6,
                c_puct=1.28,
                prior_mix=0.64,
                value_mix=0.82,
            )
        if timeout_edge > 0.0:
            return SearchProfile(
                iterations=max(self.base_iterations - 24, 28),
                max_depth=max(self.base_depth - 1, 3),
                root_action_limit=min(bundle_count, 14),
                child_action_limit=6,
                c_puct=1.08,
                prior_mix=0.74,
                value_mix=0.84,
            )
        return SearchProfile(
            iterations=max(self.base_iterations - 12, 40),
            max_depth=self.base_depth,
            root_action_limit=min(bundle_count, 12),
            child_action_limit=6,
            c_puct=1.2,
            prior_mix=0.68,
            value_mix=0.78,
        )

    def _timeout_edge(self, state: BackendState, player: int) -> float:
        enemy = 1 - player
        edge = float(state.bases[player].hp - state.bases[enemy].hp) * 10.0
        edge += float(state.die_count[player] - state.die_count[enemy]) * 1.5
        edge += float(state.super_weapon_usage[enemy] - state.super_weapon_usage[player]) * 1.0
        if hasattr(state, "ai_time"):
            edge += float(state.ai_time[enemy] - state.ai_time[player]) * 0.05
        return edge

    def _tag_bonus(self, state: BackendState, player: int, bundle: ActionBundle, round_index: int) -> float:
        tags = set(bundle.tags)
        timeout_edge = self._timeout_edge(state, player)
        first_storm_pending = state.super_weapon_usage[player] <= 0 and round_index < 115
        if round_index < 80:
            if first_storm_pending and "storm" in tags:
                return 8.0
            if first_storm_pending and "weapon" in tags:
                return -12.0
            if first_storm_pending and "build" in tags and state.tower_count(player) >= 2:
                return -5.0
            if "sell" in tags:
                return -4.0
            if "base" in tags:
                return 5.0
            if "build" in tags:
                return 3.0
            if "weapon" in tags:
                return -2.0
            return 0.0
        if round_index < 220:
            if first_storm_pending and "weapon" in tags and "storm" not in tags:
                return -10.0
            if "sell" in tags and state.super_weapon_usage[player] <= 0:
                return -3.0
            if "upgrade" in tags:
                return 2.5
            if "combo" in tags:
                return 1.5
            if "weapon" in tags:
                return 1.0
            return 0.0
        if timeout_edge < 0.0:
            if "weapon" in tags:
                return 6.0
            if "combo" in tags:
                return 4.0
            if "upgrade" in tags:
                return 2.0
            if "sell" in tags:
                return -5.0
            return 0.0
        if "weapon" in tags:
            return 4.0
        if "sell" in tags:
            return -3.0
        if "base" in tags:
            return -1.5
        if timeout_edge > 0.0 and "build" in tags:
            return 2.0
        return 0.0

    def _bundle_priority_score(self, state: BackendState, player: int, bundle: ActionBundle) -> float:
        round_index = state.round_index
        score = bundle.score + self._tag_bonus(state, player, bundle, round_index)
        tags = set(bundle.tags)
        enemy = 1 - player
        enemy_hp = state.bases[enemy].hp
        timeout_edge = self._timeout_edge(state, player)
        if enemy_hp <= 12 and ("weapon" in tags or "combo" in tags):
            score += 8.0
        if enemy_hp <= 8 and "upgrade" in tags:
            score += 3.0
        if round_index >= 96 and timeout_edge < 0.0:
            if "weapon" in tags:
                score += 7.0
            if "combo" in tags:
                score += 5.0
            if "base" in tags:
                score -= 2.5
        if round_index >= 96 and timeout_edge > 0.0:
            if "sell" in tags:
                score -= 4.0
            if "build" in tags:
                score += 1.5
        return score

    def _opening_storm_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        storm_cost = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
        storm_cd = state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM]
        if state.round_index > 118 or storm_cd > 0 or state.super_weapon_usage[player] > 0:
            return None
        if state.coins[player] >= storm_cost:
            weapon = max(
                (bundle for bundle in bundles if "storm" in bundle.tags),
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if weapon is not None:
                return weapon
            target = self._opening_storm_target(state, player)
            return ActionBundle(
                name="opening-storm",
                operations=(Operation(OperationType.USE_LIGHTNING_STORM, target[1], target[2]),),
                score=90.0,
                tags=("weapon", "storm", "override"),
            )
        cashout = self._cashout_storm_override(state, player)
        if cashout is not None:
            return cashout
        enemy_front = state.nearest_ant_distance(player)
        hp_edge = state.bases[player].hp - state.bases[1 - player].hp
        critical_defense = enemy_front <= 1 or hp_edge < -18
        if (
            10 <= state.round_index <= 118
            and not critical_defense
            and (state.coins[player] >= 35 or state.tower_count(player) >= 2)
        ):
            return ActionBundle(name="reserve-storm", score=5.0, tags=("noop", "reserve"))
        return None

    def _opening_storm_target(self, state: BackendState, player: int) -> tuple[int, int, int]:
        enemy = 1 - player
        enemy_base = PLAYER_BASES[enemy]
        candidates: list[tuple[float, int, int]] = []
        for ant in state.ants_of(enemy):
            distance_to_my_base = hex_distance(ant.x, ant.y, *PLAYER_BASES[player])
            distance_to_enemy_base = hex_distance(ant.x, ant.y, *enemy_base)
            score = 14.0 - distance_to_my_base * 1.2 + ant.level * 4.0 + max(0.0, 6 - distance_to_enemy_base) * 0.3
            candidates.append((score, ant.x, ant.y))
        for tower in state.towers_of(enemy):
            distance_to_enemy_base = hex_distance(tower.x, tower.y, *enemy_base)
            score = 5.0 + tower.level * 5.0 + max(0.0, 8 - distance_to_enemy_base) * 0.8
            candidates.append((score, tower.x, tower.y))
        if not candidates:
            return (0, enemy_base[0], enemy_base[1])
        best = max(candidates, key=lambda item: item[0])
        return (0, best[1], best[2])

    def _cashout_storm_override(
        self,
        state: BackendState,
        player: int,
        *,
        opening: bool = True,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle | None:
        storm_cost = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
        storm_cd = state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM]
        if storm_cd > 0:
            return None
        if opening:
            if not (22 <= state.round_index <= 110) or state.super_weapon_usage[player] > 0:
                return None
            min_cash = 42
            max_sells = 4
        else:
            if not (55 <= state.round_index <= 340) or state.super_weapon_usage[player] <= 0:
                return None
            min_cash = 38
            max_sells = 6
        if state.coins[player] < min_cash:
            return None
        nearest = state.nearest_ant_distance(player)
        if opening and nearest <= 2:
            return None
        if not opening and nearest <= 1 and state.bases[player].hp < state.bases[1 - player].hp - 10:
            return None

        towers = list(state.towers_of(player))
        if not towers:
            return None
        enemy = 1 - player
        enemy_front = state.nearest_ant_distance(player)
        hp_edge = state.bases[player].hp - state.bases[enemy].hp
        keep_min = 1 if enemy_front <= 4 or hp_edge < -14 else 0
        projected_cashout = state.coins[player] + sum(state.operation_income(player, Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)) for tower in towers)
        if not opening and hp_edge >= -8 and state.coins[player] >= 55 and projected_cashout >= storm_cost:
            keep_min = 0
        if len(towers) <= keep_min:
            return None

        def tower_liquidation_score(tower) -> float:
            pressure = self.catalog._local_enemy_pressure(state, player, tower.x, tower.y)
            slot = state.slot_priority(player, tower.x, tower.y)
            producer_penalty = 12.0 if tower.tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC) else 0.0
            if not opening and state.coins[player] >= 68:
                producer_penalty *= 0.45
            return pressure * 4.5 + slot * 0.7 + tower.level * 6.5 + producer_penalty

        ordered = sorted(towers, key=tower_liquidation_score)
        sell_ops: list[Operation] = []
        trial = state.clone()
        for tower in ordered:
            if len(sell_ops) >= max_sells:
                break
            if len(list(trial.towers_of(player))) <= keep_min:
                break
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not trial.can_apply_operation(player, op):
                continue
            invalid = trial.apply_operation_list(player, (op,))
            if invalid:
                continue
            sell_ops.append(op)
            if trial.coins[player] >= storm_cost:
                storm_bundle = None
                if bundles:
                    storm_bundle = max(
                        (bundle for bundle in bundles if "storm" in bundle.tags),
                        key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                        default=None,
                    )
                if storm_bundle is not None and storm_bundle.operations:
                    storm = storm_bundle.operations[-1]
                else:
                    _, x, y = self._opening_storm_target(trial, player)
                    storm = Operation(OperationType.USE_LIGHTNING_STORM, x, y)
                verify = state.clone()
                operations = tuple(sell_ops + [storm])
                if not verify.apply_operation_list(player, operations):
                    return ActionBundle(
                        name="cashout-opening-storm" if opening else "cashout-rolling-storm",
                        operations=operations,
                        score=126.0 if opening else 112.0,
                        tags=("combo", "sell", "weapon", "storm", "cashout"),
                    )
        return None

    def _rolling_storm_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        if state.round_index < 55 or state.super_weapon_usage[player] <= 0:
            return None
        storm_type = SuperWeaponType.LIGHTNING_STORM
        storm_cost = SUPER_WEAPON_STATS[storm_type].cost
        if state.weapon_cooldowns[player][storm_type] > 0:
            return None

        enemy = 1 - player
        enemy_front = state.nearest_ant_distance(player)
        own_front = state.frontline_distance(player)
        hp_edge = state.bases[player].hp - state.bases[enemy].hp
        storm_window = (
            state.round_index >= 72
            or enemy_front <= 6
            or own_front <= 8
            or hp_edge < 8
            or state.bases[enemy].hp <= 35
        )
        if not storm_window:
            return None

        if state.coins[player] >= storm_cost:
            weapon = max(
                (bundle for bundle in bundles if "storm" in bundle.tags),
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if weapon is not None:
                return weapon
            _, x, y = self._opening_storm_target(state, player)
            return ActionBundle(
                name="rolling-storm",
                operations=(Operation(OperationType.USE_LIGHTNING_STORM, x, y),),
                score=96.0,
                tags=("weapon", "storm", "override"),
            )

        if state.round_index >= 64 and state.coins[player] >= 45:
            cashout = self._cashout_storm_override(state, player, opening=False, bundles=bundles)
            if cashout is not None:
                return cashout
        return None

    def _tower_tech_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        if state.round_index < 55 or state.coins[player] < 60:
            return None
        first_storm_done = state.super_weapon_usage[player] > 0 or state.round_index >= 122
        if not first_storm_done:
            return None
        storm_cd = state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM]
        if (
            state.round_index < 260
            and state.super_weapon_usage[player] > 0
            and storm_cd <= 8
            and 72 <= state.coins[player] < SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
            and state.nearest_ant_distance(player) > 2
        ):
            return None

        upgrades = [bundle for bundle in bundles if "upgrade" in bundle.tags]
        if not upgrades:
            return None
        towers = list(state.towers_of(player))
        producer_count = sum(1 for tower in towers if tower.tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC))
        enemy_front = state.nearest_ant_distance(player)
        storm_cashout_ready = (
            state.super_weapon_usage[player] > 0
            and storm_cd == 0
            and state.coins[player] >= 55
            and enemy_front > 1
            and state.bases[player].hp >= state.bases[1 - player].hp - 10
        )
        if producer_count < 2 and enemy_front > 3 and state.coins[player] >= 60 and not storm_cashout_ready:
            producer = max(
                (bundle for bundle in upgrades if f"tower:{int(TowerType.PRODUCER)}" in bundle.tags),
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if producer is not None:
                return producer

        if producer_count >= 1 and state.round_index >= 92 and enemy_front > 3 and state.coins[player] >= 60:
            producer_branches = {
                f"tower:{int(TowerType.PRODUCER_FAST)}",
                f"tower:{int(TowerType.PRODUCER_SIEGE)}",
                f"tower:{int(TowerType.PRODUCER_MEDIC)}",
            }
            producer_upgrade = max(
                (bundle for bundle in upgrades if producer_branches & set(bundle.tags)),
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if producer_upgrade is not None:
                return producer_upgrade

        if enemy_front <= 6:
            defensive_types = {
                f"tower:{int(TowerType.ICE)}",
                f"tower:{int(TowerType.PULSE)}",
                f"tower:{int(TowerType.MORTAR_PLUS)}",
                f"tower:{int(TowerType.QUICK_PLUS)}",
                f"tower:{int(TowerType.DOUBLE)}",
            }
            defensive = max(
                (bundle for bundle in upgrades if defensive_types & set(bundle.tags)),
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if defensive is not None:
                return defensive

        if state.round_index >= 145 and sum(1 for tower in towers if tower.tower_type != TowerType.BASIC) < 2:
            return max(upgrades, key=lambda bundle: self._bundle_priority_score(state, player, bundle))
        return None

    def _tower_rebuild_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        if state.round_index < 48 or state.coins[player] < 15:
            return None
        first_storm_done = state.super_weapon_usage[player] > 0 or state.round_index >= 122
        if not first_storm_done:
            return None
        enemy_front = state.nearest_ant_distance(player)
        tower_count = state.tower_count(player)
        storm_cd = state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM]
        storm_cost = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
        if storm_cd <= 8 and state.coins[player] >= 72 and enemy_front > 3:
            return None
        if (
            tower_count == 1
            and state.coins[player] >= 60
            and storm_cd >= 10
            and enemy_front > 3
        ):
            return None
        if tower_count >= 2 and not (enemy_front <= 4 and state.coins[player] >= state.build_tower_cost(tower_count)):
            return None
        if state.coins[player] >= storm_cost and storm_cd == 0:
            return None
        builds = [bundle for bundle in bundles if "build" in bundle.tags]
        if not builds:
            return None
        if tower_count == 0 or enemy_front <= 5 or (storm_cd >= 14 and state.coins[player] >= state.build_tower_cost(tower_count)):
            return max(builds, key=lambda bundle: self._bundle_priority_score(state, player, bundle))
        return None

    def _base_upgrade_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        if state.round_index < 110 or state.round_index > 360:
            return None
        if state.coins[player] < BASE_UPGRADE_COST[0]:
            return None

        enemy = 1 - player
        first_storm_done = state.super_weapon_usage[player] > 0 or state.round_index >= 155
        safe_to_invest = state.nearest_ant_distance(player) > 4 and state.bases[player].hp >= state.bases[enemy].hp - 12
        if not first_storm_done or not safe_to_invest:
            return None
        if (
            state.round_index < 280
            and state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM] == 0
            and state.coins[player] >= 60
        ):
            return None

        upgrade_ant = next(
            (
                bundle
                for bundle in bundles
                if bundle.operations and bundle.operations[0].op_type == OperationType.UPGRADE_GENERATED_ANT
            ),
            None,
        )
        upgrade_gen = next(
            (
                bundle
                for bundle in bundles
                if bundle.operations and bundle.operations[0].op_type == OperationType.UPGRADE_GENERATION_SPEED
            ),
            None,
        )
        if state.bases[player].ant_level == 0 and upgrade_ant is not None:
            return upgrade_ant
        if state.round_index >= 170 and state.bases[player].generation_level == 0 and upgrade_gen is not None:
            return upgrade_gen
        if state.round_index >= 230 and upgrade_ant is not None:
            return upgrade_ant
        if state.round_index >= 250 and upgrade_gen is not None:
            return upgrade_gen
        return None

    def _enemy_advanced_tower_count(self, state: BackendState, player: int) -> int:
        return sum(1 for tower in state.towers_of(1 - player) if tower.level >= 2)

    def _support_weapon_override(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> ActionBundle | None:
        if state.round_index < 55:
            return None
        if state.super_weapon_usage[player] <= 0:
            return None

        enemy_front = state.nearest_ant_distance(player)
        own_front = state.frontline_distance(player)
        timeout_edge = self._timeout_edge(state, player)

        def best_with_tag(tag: str) -> ActionBundle | None:
            tagged = [bundle for bundle in bundles if tag in bundle.tags]
            if not tagged:
                return None
            return max(tagged, key=lambda bundle: self._bundle_priority_score(state, player, bundle))

        storm_cd = state.weapon_cooldowns[player][SuperWeaponType.LIGHTNING_STORM]
        storm_cost = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
        support_cost_floor = SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost
        safe_support_bank = state.coins[player] >= storm_cost + support_cost_floor or (storm_cd >= 24 and state.coins[player] >= 78)

        # EMP is a lane breaker: only spend it when our ants are actually trying to cross advanced towers.
        enemy_base = PLAYER_BASES[1 - player]
        own_forward_ants = [
            ant
            for ant in state.ants_of(player)
            if hex_distance(ant.x, ant.y, *enemy_base) <= 5
        ]
        combat_forward = sum(1 for ant in own_forward_ants if ant.kind == AntKind.COMBAT)
        if safe_support_bank and state.round_index >= 105 and self._enemy_advanced_tower_count(state, player) >= 1 and len(own_forward_ants) >= 2:
            emp = best_with_tag("emp")
            if emp is not None and self._bundle_priority_score(state, player, emp) >= 14.0:
                return emp

        # Deflector/Evasion are finish-window tools, not a replacement for the next lightning cycle.
        if safe_support_bank and own_front <= 4 and (len(own_forward_ants) >= 2 or combat_forward >= 1):
            shield = best_with_tag("shield")
            panic = best_with_tag("panic")
            support = max(
                [bundle for bundle in (shield, panic) if bundle is not None],
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if support is not None and self._bundle_priority_score(state, player, support) >= 9.0:
                return support

        if safe_support_bank and (enemy_front <= 3 or timeout_edge < -100.0):
            defensive = max(
                [
                    bundle
                    for bundle in bundles
                    if {"emp", "shield", "panic"} & set(bundle.tags)
                ],
                key=lambda bundle: self._bundle_priority_score(state, player, bundle),
                default=None,
            )
            if defensive is not None:
                return defensive
        return None

    def _should_force_fast_finish(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> bool:
        timeout_edge = self._timeout_edge(state, player)
        total_ants = len(state.ants)
        total_towers = len(state.towers)
        if state.round_index >= 220:
            return True
        if state.round_index >= 180 and (len(bundles) >= 12 or total_ants >= 14 or total_towers >= 8):
            return True
        if state.round_index >= 140 and timeout_edge > 120.0:
            return True
        return False

    def _shortlist_bundles(self, state: BackendState, player: int, bundles: list[ActionBundle]) -> list[ActionBundle]:
        if len(bundles) <= self.shortlist_size:
            return bundles

        round_index = state.round_index
        hold_bundle = bundles[0]
        rest = bundles[1:]
        ranked = sorted(
            rest,
            key=lambda bundle: self._bundle_priority_score(state, player, bundle),
            reverse=True,
        )

        keep: list[ActionBundle] = [hold_bundle]
        keep.extend(ranked[: self.shortlist_size - 1])

        # Always keep the strongest weapon line in the shortlist if one exists.
        weapon_bundle = next((bundle for bundle in ranked if "weapon" in bundle.tags), None)
        if weapon_bundle is not None and weapon_bundle not in keep:
            keep[-1] = weapon_bundle

        offense_bundle = next(
            (
                bundle
                for bundle in ranked
                if {"combo", "weapon", "upgrade", "offense"} & set(bundle.tags)
            ),
            None,
        )
        if offense_bundle is not None and offense_bundle not in keep:
            keep[-1] = offense_bundle

        return keep

    def list_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        bundles = self.catalog.build(
            state,
            player,
            context=DecisionContext.for_player(player),
            rerank=False,
        )
        return self._shortlist_bundles(state, player, bundles)

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))

        bundles = self._shortlist_bundles(state, player, bundles)
        override = self._opening_storm_override(state, player, bundles)
        if override is not None:
            return override
        override = self._rolling_storm_override(state, player, bundles)
        if override is not None:
            return override
        override = self._support_weapon_override(state, player, bundles)
        if override is not None:
            return override
        override = self._tower_rebuild_override(state, player, bundles)
        if override is not None:
            return override
        override = self._tower_tech_override(state, player, bundles)
        if override is not None:
            return override
        override = self._base_upgrade_override(state, player, bundles)
        if override is not None:
            return override
        if self._should_force_fast_finish(state, player, bundles):
            return max(bundles[1:] or bundles, key=lambda bundle: self._bundle_priority_score(state, player, bundle))
        profile = self._search_profile(state, player, bundles)
        config = self.search.search_config
        config.iterations = profile.iterations
        config.max_depth = profile.max_depth
        config.root_action_limit = profile.root_action_limit
        config.child_action_limit = profile.child_action_limit
        config.c_puct = profile.c_puct
        config.prior_mix = profile.prior_mix
        config.value_mix = profile.value_mix

        try:
            result = self.search.search(
                state=state,
                player=player,
                bundles=bundles,
                context=DecisionContext.for_player(player),
                temperature=1e-6,
                add_root_noise=False,
            )
        except Exception as exc:
            print(f"[mcts] search failed: {exc}", file=sys.stderr)
            return max(bundles[1:] or bundles, key=lambda bundle: self._bundle_priority_score(state, player, bundle))
        timeout_edge = self._timeout_edge(state, player)
        if state.round_index >= 96 and timeout_edge < 0.0 and not result.bundle.operations:
            return max(bundles[1:] or bundles, key=lambda bundle: self._bundle_priority_score(state, player, bundle))
        return result.bundle


class AI(MCTSAgent):
    def create_session(self):
        if os.getenv("AGENT_TRADITION_FORCE_MCTS") == "1":
            try:
                from protocol import ProtocolSession
            except ModuleNotFoundError as exc:
                if exc.name != "protocol":
                    raise
                from AI.protocol import ProtocolSession
            return ProtocolSession(self)

        try:
            from ai_greedy import AI as GreedyAI
            return GreedyAI().create_session()
        except Exception as exc:
            print(f"[mcts] greedy session unavailable, falling back to MCTS: {exc}", file=sys.stderr)
            try:
                from protocol import ProtocolSession
            except ModuleNotFoundError as import_exc:
                if import_exc.name != "protocol":
                    raise
                from AI.protocol import ProtocolSession
            return ProtocolSession(self)
