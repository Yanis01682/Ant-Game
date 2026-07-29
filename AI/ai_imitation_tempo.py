from __future__ import annotations

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.backend.model import Operation, Tower
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import (
    BASE_UPGRADE_COST,
    HIGHLAND_CELLS,
    OperationType,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TowerType,
)
from SDK.utils.geometry import hex_distance


PRODUCER_TYPES = {
    TowerType.PRODUCER,
    TowerType.PRODUCER_FAST,
    TowerType.PRODUCER_SIEGE,
    TowerType.PRODUCER_MEDIC,
}

ADVANCED_TYPES = {
    TowerType.HEAVY,
    TowerType.QUICK,
    TowerType.MORTAR,
    TowerType.HEAVY_PLUS,
    TowerType.ICE,
    TowerType.BEWITCH,
    TowerType.QUICK_PLUS,
    TowerType.DOUBLE,
    TowerType.SNIPER,
    TowerType.MORTAR_PLUS,
    TowerType.PULSE,
    TowerType.MISSILE,
}

STORM_COST = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
STORM_RANGE = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].attack_range
DEFLECTOR_COST = SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost
EVASION_COST = SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost
EMP_COST = SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost

OPENING_RESERVE_ROUND = 8
OPENING_RESERVE_UNTIL = 118
OPENING_RESERVE_COIN = 35
FIRST_STORM_LOCK_ROUND = 12
FIRST_STORM_BUILD_COIN_CAP = 24
STORM_CASHOUT_MIN_COIN = 24
ROLLING_STORM_ROUND = 55
ROLLING_STORM_CASHOUT_COIN = 34
TARGET_STORM_GAP = 35
LOW_HP_STORM_GAP = 35

PRODUCER_START_ROUND = 42
PRODUCER_FORCE_ROUND = 70
PRODUCER_CYCLE_ROUND = 92
PRODUCER_CASHOUT_ROUND = 64
PRODUCER_MAX_LIVE = 1
EARLY_PRODUCER_ROUND = 58
EARLY_PRODUCER_BANK = 76
PRODUCER_BANK_PULSE_ROUND = 115
PRODUCER_BANK_PULSE_COIN = 88
LIQUIDITY_ROUND = 82
LIQUIDITY_TARGET_COIN = 84
LOW_TOWER_TARGET = 2
MAX_TOWER_SOFT_CAP = 2
PRODUCER_SEED_RESERVE_ROUND = 34
PRODUCER_SEED_RESERVE_COIN = 26
STORM_SAVE_COOLDOWN = 10
STORM_SAVE_COIN = 42
MAX_TURN_OPS = 7

BASE_UPGRADE_ROUND = 190
BASE_UPGRADE_BANK = 220


def _op_key(operation: Operation) -> tuple[int, int, int]:
    return int(operation.op_type), int(operation.arg0), int(operation.arg1)


class ImitationTempoAgent(BaseAgent):
    """A low-search controller shaped from winning replay action grammar."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.last_storm_round = -999
        self.self_storm_count = 0
        self.last_support_round = -999
        self.producer_ops_seen = 0
        self.last_build_round = -999
        self.last_down_round = -999
        self.producer_upgrade_round: dict[int, int] = {}

    def on_match_start(self, player: int, seed: int) -> None:
        super().on_match_start(player, seed)
        self.last_storm_round = -999
        self.self_storm_count = 0
        self.last_support_round = -999
        self.producer_ops_seen = 0
        self.last_build_round = -999
        self.last_down_round = -999
        self.producer_upgrade_round = {}

    def on_self_operations(self, operations) -> None:
        for operation in operations:
            if operation.op_type == OperationType.USE_LIGHTNING_STORM:
                self.last_storm_round = getattr(self, "_last_round_seen", self.last_storm_round)
                self.self_storm_count += 1
            elif operation.op_type == OperationType.BUILD_TOWER:
                self.last_build_round = getattr(self, "_last_round_seen", self.last_build_round)
            elif operation.op_type == OperationType.DOWNGRADE_TOWER:
                self.last_down_round = getattr(self, "_last_round_seen", self.last_down_round)
                self.producer_upgrade_round.pop(operation.arg0, None)
            elif operation.op_type == OperationType.UPGRADE_TOWER and self._tower_type_or_none(operation.arg1) in PRODUCER_TYPES:
                self.producer_ops_seen += 1
                self.producer_upgrade_round[operation.arg0] = getattr(self, "_last_round_seen", 0)
            elif operation.op_type in (OperationType.USE_DEFLECTOR, OperationType.USE_EMERGENCY_EVASION):
                self.last_support_round = getattr(self, "_last_round_seen", self.last_support_round)

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        del bundles
        self._last_round_seen = state.round_index
        operations = self._choose_operations(state, player)
        return ActionBundle(name="imitation-tempo", operations=tuple(operations), score=0.0, tags=("tempo",))

    def _choose_operations(self, state: BackendState, player: int) -> list[Operation]:
        plan: list[Operation] = []
        force_first_producer = self._should_force_first_producer(state, player)

        if force_first_producer:
            self._append_from_chooser(state, player, plan, self._try_producer_cycle)
            self._append_from_chooser(state, player, plan, self._try_producer_seed)

        self._append_from_chooser(state, player, plan, self._try_storm_tempo)

        if not self._has_operation(plan, OperationType.USE_LIGHTNING_STORM):
            # Mature producers are treated as tempo cash, not permanent assets.
            self._append_from_chooser(state, player, plan, self._try_producer_cycle)
            self._append_from_chooser(state, player, plan, self._try_liquidity_cashout)
            self._append_from_chooser(state, player, plan, self._try_storm_tempo)

        if not force_first_producer:
            self._append_from_chooser(state, player, plan, self._try_early_producer_commit)
            self._append_from_chooser(state, player, plan, self._try_producer_cycle)
            self._append_from_chooser(state, player, plan, self._try_producer_seed)
            self._append_from_chooser(state, player, plan, self._try_producer_bank_pulse)
            self._append_from_chooser(state, player, plan, self._try_liquidity_cashout)

        if not self._storm_saving_mode(state, player, self._projected_coin(state, player, plan)):
            self._append_from_chooser(state, player, plan, self._try_support_window)
            self._append_from_chooser(state, player, plan, self._try_producer_seed)
            self._append_from_chooser(state, player, plan, self._try_low_tower_rebuild)
            self._append_from_chooser(state, player, plan, self._try_defensive_upgrade)

        self._append_from_chooser(state, player, plan, self._try_cash_trim)
        self._append_from_chooser(state, player, plan, self._try_base_upgrade)
        return self._legal_prefix(state, player, plan[:MAX_TURN_OPS])

    def _append_from_chooser(self, state: BackendState, player: int, plan: list[Operation], chooser) -> bool:
        if len(plan) >= MAX_TURN_OPS:
            return False
        before = len(plan)
        op = chooser(state, player, plan)
        if op is None:
            return len(plan) > before
        if not state.can_apply_operation(player, op, plan):
            del plan[before:]
            return False
        plan.append(op)
        if len(plan) > MAX_TURN_OPS:
            del plan[MAX_TURN_OPS:]
        return True

    def _has_operation(self, plan: list[Operation], op_type: OperationType) -> bool:
        return any(operation.op_type == op_type for operation in plan)

    def _legal_prefix(self, state: BackendState, player: int, plan: list[Operation]) -> list[Operation]:
        accepted: list[Operation] = []
        for operation in plan:
            if state.can_apply_operation(player, operation, accepted):
                accepted.append(operation)
        return accepted

    def _should_force_first_producer(self, state: BackendState, player: int) -> bool:
        if self.self_storm_count <= 0 or self.producer_ops_seen > 0:
            return False
        if not (38 <= state.round_index <= 118):
            return False
        if self._enemy_front(state, player) < 3 or self._hp_edge(state, player) < -14:
            return False
        storm_ready = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0
        if storm_ready and state.coins[player] >= STORM_COST:
            return False
        if state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] <= 5 and state.coins[player] >= 72:
            return False
        return True

    def _projected_coin(self, state: BackendState, player: int, plan: list[Operation]) -> int:
        coin = state.coins[player]
        tower_count = state.tower_count(player)
        for operation in plan:
            if operation.op_type == OperationType.DOWNGRADE_TOWER:
                tower = state.tower_by_id(operation.arg0)
                if tower is None:
                    continue
                coin += state.operation_income(player, operation, tower_count)
                if tower.tower_type == TowerType.BASIC:
                    tower_count = max(0, tower_count - 1)
            elif operation.op_type == OperationType.BUILD_TOWER:
                coin -= state.build_tower_cost(tower_count)
                tower_count += 1
            elif operation.op_type == OperationType.UPGRADE_TOWER:
                target_type = self._tower_type_or_none(operation.arg1)
                if target_type is not None:
                    coin -= state.upgrade_tower_cost(target_type)
            elif operation.op_type in (
                OperationType.USE_LIGHTNING_STORM,
                OperationType.USE_EMP_BLASTER,
                OperationType.USE_DEFLECTOR,
                OperationType.USE_EMERGENCY_EVASION,
            ):
                coin -= state.weapon_cost(SuperWeaponType(operation.op_type % 10))
            elif operation.op_type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED):
                base = state.bases[player]
                level = base.ant_level if operation.op_type == OperationType.UPGRADE_GENERATED_ANT else base.generation_level
                coin -= state.upgrade_base_cost(level)
        return coin

    def _live_towers(self, state: BackendState, player: int) -> list[Tower]:
        return [tower for tower in state.towers_of(player) if tower.hp > 0]

    def _producer_towers(self, state: BackendState, player: int) -> list[Tower]:
        return [tower for tower in self._live_towers(state, player) if tower.tower_type in PRODUCER_TYPES]

    def _tower_type_or_none(self, value: int) -> TowerType | None:
        try:
            return TowerType(value)
        except ValueError:
            return None

    def _ant_kind_value(self, ant) -> int:
        kind = getattr(ant, "kind", 0)
        try:
            return int(kind)
        except (TypeError, ValueError):
            return 0

    def _enemy_front(self, state: BackendState, player: int) -> int:
        return state.nearest_ant_distance(player)

    def _own_front(self, state: BackendState, player: int) -> int:
        return state.frontline_distance(player)

    def _hp_edge(self, state: BackendState, player: int) -> int:
        return state.bases[player].hp - state.bases[1 - player].hp

    def _storm_gap_ready(self, state: BackendState, player: int) -> bool:
        if self.self_storm_count == 0:
            return True
        gap = state.round_index - self.last_storm_round
        target = LOW_HP_STORM_GAP if self._hp_edge(state, player) < -8 else TARGET_STORM_GAP
        return gap >= target

    def _storm_saving_mode(self, state: BackendState, player: int, coin: int) -> bool:
        if self.self_storm_count <= 0:
            return False
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if storm_cd <= STORM_SAVE_COOLDOWN and coin >= STORM_SAVE_COIN:
            return True
        if state.round_index - self.last_storm_round >= TARGET_STORM_GAP - 4 and coin >= STORM_SAVE_COIN:
            return True
        return False

    def _try_storm_tempo(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < 18:
            return None
        if state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] > 0:
            return None
        if not self._storm_gap_ready(state, player):
            return None
        if self._enemy_front(state, player) <= 1 and state.bases[player].hp <= 16:
            return None

        coin = self._projected_coin(state, player, plan)
        if coin < STORM_COST:
            should_cashout = (
                state.round_index <= OPENING_RESERVE_UNTIL
                and coin >= STORM_CASHOUT_MIN_COIN
                and self.self_storm_count == 0
            ) or (
                state.round_index >= ROLLING_STORM_ROUND
                and coin >= ROLLING_STORM_CASHOUT_COIN
                and self.self_storm_count > 0
            )
            if should_cashout:
                preserve_seed = self.self_storm_count == 0 and state.round_index < PRODUCER_FORCE_ROUND
                sell_plan = self._cashout_for_coin(
                    state,
                    player,
                    STORM_COST,
                    plan,
                    max_ops=5,
                    preserve_seed=preserve_seed,
                )
                if sell_plan:
                    plan.extend(sell_plan)
                    coin = self._projected_coin(state, player, plan)
        if coin < STORM_COST:
            return None

        x, y = self._storm_target(state, player)
        return Operation(OperationType.USE_LIGHTNING_STORM, x, y)

    def _storm_target(self, state: BackendState, player: int) -> tuple[int, int]:
        enemy = 1 - player
        enemy_base = PLAYER_BASES[enemy]
        candidates: set[tuple[int, int]] = {enemy_base}
        candidates.update((ant.x, ant.y) for ant in state.ants_of(enemy))
        candidates.update((tower.x, tower.y) for tower in state.towers_of(enemy))
        best = enemy_base
        best_score = -1e9
        for x, y in candidates:
            score = 0.0
            for ant in state.ants_of(enemy):
                gap = hex_distance(x, y, ant.x, ant.y)
                if gap <= STORM_RANGE:
                    score += 14.0 + ant.level * 6.0 + max(0, 3 - gap) * 1.5
                    score += 4.0 if self._ant_kind_value(ant) == 1 else 0.0
            for tower in state.towers_of(enemy):
                gap = hex_distance(x, y, tower.x, tower.y)
                if gap <= STORM_RANGE:
                    score += 8.0 + tower.level * 5.0
                    if tower.tower_type in PRODUCER_TYPES:
                        score += 10.0
            base_gap = hex_distance(x, y, *enemy_base)
            score += max(0, 8 - base_gap) * 2.0
            if base_gap <= STORM_RANGE:
                score += 18.0
            own_gap = hex_distance(x, y, *PLAYER_BASES[player])
            if own_gap <= STORM_RANGE and base_gap > 7:
                score -= 18.0
            if score > best_score:
                best = (x, y)
                best_score = score
        return best

    def _cashout_for_coin(
        self,
        state: BackendState,
        player: int,
        target_coin: int,
        plan: list[Operation] | None = None,
        *,
        max_ops: int,
        preserve_seed: bool = True,
    ) -> list[Operation]:
        base_plan = plan or []
        coin = self._projected_coin(state, player, base_plan)
        tower_count = state.tower_count(player)
        for operation in base_plan:
            if operation.op_type == OperationType.DOWNGRADE_TOWER:
                tower = state.tower_by_id(operation.arg0)
                if tower is not None and tower.tower_type == TowerType.BASIC:
                    tower_count = max(0, tower_count - 1)
            elif operation.op_type == OperationType.BUILD_TOWER:
                tower_count += 1
        ops: list[Operation] = []
        for tower in self._sell_order(state, player, preserve_seed=preserve_seed):
            if coin >= target_coin or len(ops) >= max_ops:
                break
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op, base_plan + ops):
                continue
            ops.append(op)
            coin += state.operation_income(player, op, tower_count)
            if tower.tower_type == TowerType.BASIC:
                tower_count = max(0, tower_count - 1)
        return ops if coin >= target_coin else []

    def _sell_order(self, state: BackendState, player: int, *, preserve_seed: bool = True) -> list[Tower]:
        enemy_front = self._enemy_front(state, player)

        def score(tower: Tower) -> float:
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            refund = state.operation_income(player, Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id))
            value = refund - pressure * 22.0
            if tower.tower_type in PRODUCER_TYPES:
                value += 45.0 if self._producer_mature(state, tower) else -120.0
            elif tower.tower_type != TowerType.BASIC:
                value -= 22.0 + tower.level * 10.0
            elif preserve_seed and state.round_index >= PRODUCER_SEED_RESERVE_ROUND and self._is_safe_producer_site(state, player, tower.x, tower.y):
                value -= 55.0
            if enemy_front <= 3 and pressure > 0:
                value -= 100.0
            value -= state.slot_priority(player, tower.x, tower.y) * 0.35
            return value

        return sorted(self._live_towers(state, player), key=score, reverse=True)

    def _producer_mature(self, state: BackendState, tower: Tower) -> bool:
        if tower.tower_type not in PRODUCER_TYPES:
            return False
        return tower.cooldown_clock <= 1 or state.round_index >= PRODUCER_CYCLE_ROUND

    def _try_early_producer_commit(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < EARLY_PRODUCER_ROUND or self.producer_ops_seen > 0:
            return None
        if self.self_storm_count <= 0 or self._enemy_front(state, player) < 3 or self._hp_edge(state, player) < -12:
            return None
        if self._producer_towers(state, player):
            return None
        coin = self._projected_coin(state, player, plan)
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if storm_cd <= 8 and coin < STORM_COST + 20:
            return None
        basic = self._best_basic_for_producer(state, player)
        if basic is not None:
            if coin >= 60 and (storm_cd > 10 or coin >= STORM_COST + 60):
                return Operation(OperationType.UPGRADE_TOWER, basic.tower_id, int(TowerType.PRODUCER))
            return None
        if state.round_index - self.last_build_round < 8:
            return None
        cost = state.build_tower_cost(state.tower_count(player))
        if coin < max(EARLY_PRODUCER_BANK, cost + 28):
            return None
        pos = self._best_build_site(state, player, prefer_safe=True)
        if pos is not None:
            return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
        return None

    def _try_producer_cycle(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < PRODUCER_START_ROUND:
            return None
        if self._enemy_front(state, player) < 3:
            return None
        if self._hp_edge(state, player) < -14:
            return None
        producers = self._producer_towers(state, player)
        storm_ready = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0
        coin = self._projected_coin(state, player, plan)

        for tower in producers:
            if state.round_index >= PRODUCER_CASHOUT_ROUND and self._producer_mature(state, tower):
                if storm_ready or coin >= 52 or len(producers) > PRODUCER_MAX_LIVE:
                    return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)

        if len(producers) >= PRODUCER_MAX_LIVE:
            return None
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if self._storm_saving_mode(state, player, coin) and not (state.round_index >= PRODUCER_FORCE_ROUND and storm_cd > 12):
            return None
        if storm_ready and coin >= STORM_COST and state.round_index < PRODUCER_FORCE_ROUND:
            return None
        if coin < 60:
            return None

        basic = self._best_basic_for_producer(state, player)
        if basic is not None:
            return Operation(OperationType.UPGRADE_TOWER, basic.tower_id, int(TowerType.PRODUCER))

        return None

    def _try_producer_seed(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < PRODUCER_SEED_RESERVE_ROUND or self.self_storm_count <= 0:
            return None
        if state.round_index - self.last_build_round < 8:
            return None
        if self._enemy_front(state, player) < 3 or self._hp_edge(state, player) < -14:
            return None
        if self._producer_towers(state, player):
            return None
        if self._best_basic_for_producer(state, player) is not None:
            return None

        coin = self._projected_coin(state, player, plan)
        if self._storm_saving_mode(state, player, coin):
            return None
        cost = state.build_tower_cost(state.tower_count(player))
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if storm_cd <= 8 and coin < STORM_COST:
            return None
        seed_bank = 58 if state.round_index >= PRODUCER_FORCE_ROUND else PRODUCER_SEED_RESERVE_COIN
        if coin < max(cost, seed_bank):
            return None
        pos = self._best_build_site(state, player, prefer_safe=True)
        if pos is not None:
            return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
        return None

    def _try_producer_bank_pulse(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < PRODUCER_BANK_PULSE_ROUND:
            return None
        if self._enemy_front(state, player) <= 2 or self._hp_edge(state, player) < -10:
            return None
        if self._producer_towers(state, player):
            return None
        coin = self._projected_coin(state, player, plan)
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if storm_cd <= 8 and coin < STORM_COST + 52:
            return None
        if coin < PRODUCER_BANK_PULSE_COIN:
            return None
        basic = self._best_basic_for_producer(state, player)
        if basic is not None:
            return Operation(OperationType.UPGRADE_TOWER, basic.tower_id, int(TowerType.PRODUCER))
        if state.round_index - self.last_build_round < 8:
            return None
        if coin < PRODUCER_BANK_PULSE_COIN + state.build_tower_cost(state.tower_count(player)) + 28:
            return None
        pos = self._best_build_site(state, player, prefer_safe=True)
        if pos is not None:
            return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])
        return None

    def _try_liquidity_cashout(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < LIQUIDITY_ROUND or self.self_storm_count <= 0:
            return None
        if self._enemy_front(state, player) <= 2:
            return None
        coin = self._projected_coin(state, player, plan)
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        storm_gap = state.round_index - self.last_storm_round
        urgent_storm = storm_cd <= 8 or storm_gap >= TARGET_STORM_GAP - 6
        target = STORM_COST if urgent_storm else LIQUIDITY_TARGET_COIN
        if coin >= target:
            return None
        if not urgent_storm and state.tower_count(player) <= 1:
            return None
        for tower in self._sell_order(state, player, preserve_seed=False):
            if tower.tower_type in PRODUCER_TYPES and not self._producer_mature(state, tower):
                continue
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if state.can_apply_operation(player, op, plan):
                return op
        return None

    def _best_basic_for_producer(self, state: BackendState, player: int) -> Tower | None:
        candidates = [
            tower for tower in self._live_towers(state, player)
            if tower.tower_type == TowerType.BASIC and self._is_safe_producer_site(state, player, tower.x, tower.y)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda tower: self._producer_site_score(state, player, tower.x, tower.y))

    def _is_safe_producer_site(self, state: BackendState, player: int, x: int, y: int) -> bool:
        if self._local_enemy_pressure(state, player, x, y) > 1.5:
            return False
        return hex_distance(x, y, *PLAYER_BASES[player]) >= 3

    def _producer_site_score(self, state: BackendState, player: int, x: int, y: int) -> float:
        enemy_base = PLAYER_BASES[1 - player]
        return hex_distance(x, y, *PLAYER_BASES[player]) * 0.6 - hex_distance(x, y, *enemy_base) * 0.25 + state.slot_priority(player, x, y) * 0.1

    def _try_low_tower_rebuild(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        coin = self._projected_coin(state, player, plan)
        tower_count = state.tower_count(player)
        if tower_count >= LOW_TOWER_TARGET:
            return None
        if self.self_storm_count == 0 and state.round_index >= FIRST_STORM_LOCK_ROUND:
            if self._enemy_front(state, player) > 2 or coin >= FIRST_STORM_BUILD_COIN_CAP:
                return None
        if self._storm_saving_mode(state, player, coin):
            return None
        if self.self_storm_count > 0 and state.round_index >= PRODUCER_SEED_RESERVE_ROUND:
            if self._enemy_front(state, player) > 0:
                return None
            if state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] <= 12 and coin < STORM_COST:
                return None
            if tower_count > 0:
                return None
        if state.round_index >= OPENING_RESERVE_ROUND and self.self_storm_count == 0 and coin >= OPENING_RESERVE_COIN and self._enemy_front(state, player) > 1:
            return None
        cost = state.build_tower_cost(tower_count)
        if coin < cost:
            return None
        pos = self._best_build_site(state, player, prefer_safe=False)
        if pos is None:
            return None
        return Operation(OperationType.BUILD_TOWER, pos[0], pos[1])

    def _best_build_site(self, state: BackendState, player: int, *, prefer_safe: bool) -> tuple[int, int] | None:
        sites = list(STRATEGIC_BUILD_ORDER[player])
        sites.extend(pos for pos in HIGHLAND_CELLS[player] if pos not in sites)
        best: tuple[int, int] | None = None
        best_score = -1e9
        for x, y in sites:
            if state.tower_at(x, y) is not None:
                continue
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if not state.can_apply_operation(player, op):
                continue
            pressure = self._local_enemy_pressure(state, player, x, y)
            safe = self._is_safe_producer_site(state, player, x, y)
            score = state.slot_priority(player, x, y) + pressure * 4.0
            if prefer_safe:
                score += 30.0 if safe else -30.0
                score += self._producer_site_score(state, player, x, y)
            if score > best_score:
                best = (x, y)
                best_score = score
        return best

    def _try_defensive_upgrade(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        coin = self._projected_coin(state, player, plan)
        if coin < 60:
            return None
        if self._storm_saving_mode(state, player, coin):
            return None
        if (
            state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] <= 8
            and 65 <= coin < STORM_COST
            and self._enemy_front(state, player) > 2
        ):
            return None
        if state.tower_count(player) > MAX_TOWER_SOFT_CAP and self._enemy_front(state, player) > 4:
            return None

        best: tuple[float, Operation] | None = None
        for tower in self._live_towers(state, player):
            if tower.tower_type != TowerType.BASIC:
                continue
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            target = self._defense_target_for(tower, pressure, state, player)
            if target is None:
                continue
            op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
            if not state.can_apply_operation(player, op, plan):
                continue
            score = pressure * 6.0 + state.slot_priority(player, tower.x, tower.y) * 0.25
            if self._enemy_front(state, player) <= 4:
                score += 10.0
            if best is None or score > best[0]:
                best = (score, op)
        return best[1] if best is not None and (best[0] >= 12.0 or self._enemy_front(state, player) <= 4) else None

    def _defense_target_for(self, tower: Tower, pressure: float, state: BackendState, player: int) -> TowerType | None:
        if pressure >= 4.0:
            return TowerType.MORTAR
        if self._enemy_front(state, player) <= 4:
            return TowerType.ICE
        if pressure >= 2.0:
            return TowerType.QUICK
        return None

    def _try_cash_trim(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        coin = self._projected_coin(state, player, plan)
        if state.tower_count(player) <= MAX_TOWER_SOFT_CAP:
            return None
        if self._enemy_front(state, player) <= 3:
            return None
        if coin < 95 and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] <= 12:
            return None
        for tower in self._sell_order(state, player):
            if tower.tower_type in PRODUCER_TYPES and not self._producer_mature(state, tower):
                continue
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
        return None

    def _try_support_window(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if self.self_storm_count <= 0 or state.round_index - self.last_support_round < 22:
            return None
        coin = self._projected_coin(state, player, plan)
        storm_cd = state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
        if storm_cd <= 10 and coin < STORM_COST + DEFLECTOR_COST:
            return None
        own_front = self._own_front(state, player)
        if own_front > 4:
            return None
        forward_ants = [
            ant for ant in state.ants_of(player)
            if hex_distance(ant.x, ant.y, *PLAYER_BASES[1 - player]) <= 5
        ]
        if len(forward_ants) < 2 and coin < STORM_COST + EVASION_COST:
            return None

        if coin >= DEFLECTOR_COST and state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0:
            x, y = self._support_target(state, player, forward_ants)
            op = Operation(OperationType.USE_DEFLECTOR, x, y)
            if state.can_apply_operation(player, op, plan):
                return op
        if coin >= EVASION_COST and state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0:
            x, y = self._support_target(state, player, forward_ants)
            op = Operation(OperationType.USE_EMERGENCY_EVASION, x, y)
            if state.can_apply_operation(player, op, plan):
                return op
        if coin >= EMP_COST and state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0:
            target = self._emp_target(state, player)
            if target is not None:
                op = Operation(OperationType.USE_EMP_BLASTER, target[0], target[1])
                if state.can_apply_operation(player, op, plan):
                    return op
        return None

    def _support_target(self, state: BackendState, player: int, ants) -> tuple[int, int]:
        if not ants:
            return PLAYER_BASES[1 - player]
        best = min(ants, key=lambda ant: hex_distance(ant.x, ant.y, *PLAYER_BASES[1 - player]))
        return best.x, best.y

    def _emp_target(self, state: BackendState, player: int) -> tuple[int, int] | None:
        enemy_towers = [tower for tower in state.towers_of(1 - player) if tower.tower_type != TowerType.BASIC]
        if not enemy_towers:
            return None
        best = max(enemy_towers, key=lambda tower: tower.level * 10.0 + max(0, 8 - hex_distance(tower.x, tower.y, *PLAYER_BASES[1 - player])))
        return best.x, best.y

    def _try_base_upgrade(self, state: BackendState, player: int, plan: list[Operation]) -> Operation | None:
        if state.round_index < BASE_UPGRADE_ROUND or self._enemy_front(state, player) <= 4:
            return None
        coin = self._projected_coin(state, player, plan)
        if coin < BASE_UPGRADE_BANK:
            return None
        if state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] <= 8 and coin < BASE_UPGRADE_BANK + STORM_COST:
            return None
        base = state.bases[player]
        if base.ant_level < 1 and coin >= BASE_UPGRADE_COST[base.ant_level]:
            return Operation(OperationType.UPGRADE_GENERATED_ANT)
        if base.generation_level < 1 and coin >= BASE_UPGRADE_COST[base.generation_level]:
            return Operation(OperationType.UPGRADE_GENERATION_SPEED)
        return None

    def _local_enemy_pressure(self, state: BackendState, player: int, x: int, y: int) -> float:
        total = 0.0
        for ant in state.ants_of(1 - player):
            gap = hex_distance(x, y, ant.x, ant.y)
            if gap <= 5:
                total += max(0.0, 5.5 - gap) * (1.0 + ant.level * 0.4)
        return total


class AI(ImitationTempoAgent):
    pass
