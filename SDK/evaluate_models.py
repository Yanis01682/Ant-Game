from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.training import AlphaZeroSelfPlayTrainer, AlphaZeroTrainerConfig, AntWarSequentialEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate one or more Ant-Game checkpoints with the current logic.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to evaluate.")
    parser.add_argument("--champion-path", type=str, default="checkpoints/overnight_01_champion.npz")
    parser.add_argument("--evaluation-episodes", type=int, default=6)
    parser.add_argument("--promotion-episodes", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=96)
    parser.add_argument("--search-iterations", type=int, default=48)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-native-backend", action="store_true")
    return parser.parse_args()


def build_trainer(checkpoint: str, args: argparse.Namespace) -> AlphaZeroSelfPlayTrainer:
    config = AlphaZeroTrainerConfig(
        batches=1,
        episodes=1,
        search_iterations=args.search_iterations,
        max_depth=args.max_depth,
        max_rounds=args.max_rounds,
        max_actions=args.max_actions,
        resume_from=checkpoint,
        checkpoint_path=checkpoint,
        champion_path=args.champion_path,
        evaluation_episodes=args.evaluation_episodes,
        promotion_episodes=args.promotion_episodes,
        seed=args.seed,
    )

    def env_factory(seed: int):
        return AntWarSequentialEnv(
            seed=seed,
            max_actions=config.max_actions,
            prefer_native_backend=args.prefer_native_backend,
        )

    return AlphaZeroSelfPlayTrainer(env_factory=env_factory, config=config, logger=None)


def evaluate_checkpoint(checkpoint: str, args: argparse.Namespace) -> dict[str, object]:
    trainer = build_trainer(checkpoint, args)
    heuristic = trainer.evaluate_against_heuristic(num_episodes=args.evaluation_episodes)
    champion = trainer.evaluate_against_champion(num_episodes=args.promotion_episodes)
    return {
        "checkpoint": checkpoint,
        "champion_path": args.champion_path,
        "model_device": getattr(trainer.model, "device", "unknown"),
        "search_iterations": args.search_iterations,
        "max_depth": args.max_depth,
        "max_rounds": args.max_rounds,
        "evaluation": heuristic,
        "promotion_probe": champion,
    }


def main() -> None:
    args = parse_args()
    results = [evaluate_checkpoint(checkpoint, args) for checkpoint in args.checkpoints]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
