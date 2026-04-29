from __future__ import annotations

from SDK.training import AlphaZeroSelfPlayTrainer, AlphaZeroTrainerConfig
from SDK.backend.model import Ant
from SDK.training import AntWarSequentialEnv
from SDK.utils.constants import AntStatus, OperationType


def test_sequential_env_advances_round_only_after_player_1_acts() -> None:
    env = AntWarSequentialEnv(seed=21)
    try:
        env.reset(seed=21)
        assert env.agent_selection == "player_0"

        _, reward0, termination0, truncation0, info0 = env.last()
        assert reward0 == 0.0
        assert termination0 is False
        assert truncation0 is False
        assert info0["to_play"] == 0

        env.step(0)
        assert env.state.round_index == 0
        assert env.agent_selection == "player_1"

        _, reward1, termination1, truncation1, info1 = env.last()
        assert reward1 == 0.0
        assert termination1 is False
        assert truncation1 is False
        assert info1["to_play"] == 1

        env.step(0)
        assert env.state.round_index == 1
        assert env.agent_selection == "player_0"
    finally:
        env.close()


def test_lightning_storm_is_immediate_but_round_settlement_remains_deferred() -> None:
    env = AntWarSequentialEnv(seed=7)
    try:
        env.reset(seed=7)
        env.state.coins[0] = 200
        env.state.ants.append(Ant(999, 1, 9, 9, hp=20, level=0))
        env._capture_round_start()
        env._refresh_bundles()
        env._update_infos()

        _, reward0, termination0, truncation0, info0 = env.last()
        assert reward0 == 0.0
        assert termination0 is False
        assert truncation0 is False

        lightning_index = next(
            index
            for index, bundle in enumerate(info0["bundles"])
            if bundle.operations and bundle.operations[0].op_type == OperationType.USE_LIGHTNING_STORM
        )

        env.step(lightning_index)

        enemy = next(ant for ant in env.state.ants if ant.ant_id == 999)
        assert env.state.round_index == 0
        assert env.agent_selection == "player_1"
        assert enemy.hp == 0
        assert enemy.status == AntStatus.FAIL

        _, reward1, termination1, truncation1, info1 = env.last()
        assert reward1 == 0.0
        assert termination1 is False
        assert truncation1 is False
        assert info1["to_play"] == 1
    finally:
        env.close()


def test_capped_round_resolution_assigns_timeout_winner() -> None:
    config = AlphaZeroTrainerConfig(
        batches=1,
        episodes=1,
        max_rounds=8,
        max_actions=16,
        search_iterations=2,
        max_depth=1,
    )
    trainer = AlphaZeroSelfPlayTrainer(
        env_factory=lambda seed: AntWarSequentialEnv(seed=seed, max_actions=16),
        config=config,
        logger=None,
    )
    env = AntWarSequentialEnv(seed=13, max_actions=16)
    try:
        env.reset(seed=13)
        env.state.round_index = config.max_rounds
        env.state.terminal = False
        env.state.winner = None
        env.state.bases[0].hp = 42
        env.state.bases[1].hp = 35
        assert trainer._resolve_episode_winner(env) == 0
    finally:
        env.close()
