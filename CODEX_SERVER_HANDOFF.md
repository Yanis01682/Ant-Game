# Codex Server Handoff

## Current Branch

- Branch: `zhangzhiyuan-agent`
- Base: synced with `origin/main` as of `2026-04-26`
- Official upstream latest commit confirmed during local check:
  - `0a6bee4` at `2026-04-19 13:37:16 +0800`
  - message: `fix: mcts and training logic`

## Goal

Build a high-ranking competition AI for the Ant-Game leaderboard using:

- stronger action generation
- cleaner MCTS decision logic
- self-play training
- league-style opponents
- champion/challenger promotion
- experiment sweeps on GPU servers

## Important Context

- The user is not deeply focused on low-level implementation details and wants Codex to take ownership.
- Training is intended to run on a server with access to multiple A100 GPUs.
- Local environment had native backend / C++ build issues, so most logic work so far was done against Python-side code paths.
- The current state is designed to be run on a server environment next.

## What Was Changed

### 1. Main agent cleanup

File: `AI/ai_mcts.py`

- Replaced the previously messy double-`MCTSAgent` file with a single clean implementation.
- Kept official `PriorGuidedMCTS` as the main backbone.
- Added:
  - candidate shortlist logic
  - phase-dependent search profile tuning
  - safe patch for invalid base upgrade actions
- Agent now loads checkpoints from:
  - `AI/ai_mcts_model.npz`
  - `checkpoints/ai_mcts_latest.npz`
  - `SDK/checkpoints/ai_mcts_latest.npz`

### 2. Stronger action generation

File: `SDK/utils/actions.py`

- Removed broken duplicate rerank residue that caused syntax failure.
- Fixed one-step rerank rollout to use actual backend state APIs.
- Improved superweapon center selection:
  - `Lightning Storm`
  - `EMP`
  - `Deflector`
  - `Emergency Evasion`
- Added tower-aware `Lightning Storm` scoring.
- Expanded combo generation to include more realistic tactical pairings.

### 3. Training pipeline rewrite

File: `SDK/train_mcts.py`

- Replaced old ad hoc training scaffold with a real entrypoint around `AlphaZeroSelfPlayTrainer`.
- Added CLI flags for:
  - search
  - rounds
  - opponent pool
  - promotion gate
  - checkpoint paths
  - export path

### 4. Self-play trainer improvements

File: `SDK/training/alphazero.py`

- Self-play no longer trains only from mirror matches.
- Added opponent pool:
  - mirror self-play
  - heuristic baseline
  - historical checkpoints
  - champion checkpoint
- Added sample-level value target shaping:
  - terminal outcome
  - time discount
  - early bootstrap from search root value
- Added champion/challenger logic:
  - every batch saves `latest`
  - challenger must beat champion in promotion matches
  - only promoted model becomes champion

### 5. Logging and observability

File: `SDK/training/logging_utils.py`

- Logs now include:
  - trained side
  - opponent kind
  - promotion win rate
  - whether champion promotion occurred
  - matchup win rates in batch summaries

### 6. Sweep launcher

Files:

- `SDK/train_sweep.py`
- `SDK/train_sweep.sh`

- Added predefined experiment profiles:
  - `balanced`
  - `champion_hunter`
  - `deep_search`
  - `broad_league`
- Supports multi-profile, multi-seed experiment launching.

### 7. Packaging and runtime alignment

File: `AI/zip_mcts.sh`

- Packaging now prefers champion/export-ready checkpoint naming.
- Intended runtime model file inside packaged submission is `ai_mcts_model.npz`.

### 8. Dependencies and loop script

Files:

- `requirements.txt`
- `train_loop.sh`

- Added Python training dependencies:
  - `numpy`
  - `gymnasium`
  - `pettingzoo`
- Updated training loop script to use the new trainer.

## Current Expected Training Artifacts

- latest checkpoint:
  - `checkpoints/ai_mcts_latest.npz`
- champion checkpoint:
  - `checkpoints/ai_mcts_champion.npz`
- exported agent model:
  - `AI/ai_mcts_model.npz`
- sweep exports:
  - `AI/sweep_exports/*.npz`

## Suggested First Actions On Server

1. Verify environment:
   - Python version
   - CUDA
   - PyTorch GPU visibility
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install torch`
3. Run one dry / smoke training command.
4. Run sweep profiles on server.
5. Monitor:
   - `promotion_win_rate`
   - `matchup_heuristic_win_rate`
   - `matchup_champion_win_rate`
   - side bias metrics

## Recommended Initial Server Commands

### Single training run

```bash
bash SDK/train_mcts.sh \
  --batches 32 \
  --episodes 16 \
  --search-iterations 96 \
  --max-depth 5 \
  --max-rounds 192 \
  --opponent-pool-size 6 \
  --selfplay-mirror-ratio 0.35 \
  --selfplay-heuristic-ratio 0.25 \
  --promotion-episodes 8 \
  --promotion-win-rate 0.56 \
  --run-name league-champion-01
```

### Sweep run

```bash
bash SDK/train_sweep.sh --runs-per-profile 1
```

### Stronger sweep

```bash
bash SDK/train_sweep.sh --runs-per-profile 2
```

## What To Do Next

Priority order:

1. Run training on server and inspect actual metrics.
2. Compare sweep profiles and identify the strongest configuration.
3. Diagnose matchup weaknesses:
   - weak vs heuristic
   - weak vs champion
   - side bias
4. Adjust:
   - action generation heuristics
   - self-play ratios
   - promotion strictness
   - search depth / iterations
5. Package best champion for submission.

## Known Caveats

- Local machine lacked `gymnasium`, so full local trainer execution was not performed here.
- Local Windows environment had native backend / C++ build issues.
- Python-side tests relevant to action generation and agent integration were passing after edits.

## Minimal Prompt For Server-side Codex

Use this if needed:

> Read `CODEX_SERVER_HANDOFF.md` first, then take over this Ant-Game training pipeline on the server. Verify environment, install missing dependencies, run the recommended training/sweep commands, analyze the resulting metrics, and continue improving the strongest competition agent on branch `zhangzhiyuan-agent`.
