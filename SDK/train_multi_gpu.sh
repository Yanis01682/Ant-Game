#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="/home/zhangzhiyuan/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="antgame"
RESUME_FROM="${1:-checkpoints/overnight_01_latest.npz}"
STAMP="${2:-$(date +%Y%m%d-%H%M%S)}"

launch_run() {
  local gpu="$1"
  local session="$2"
  local checkpoint="$3"
  local champion="$4"
  local export_model="$5"
  local run_name="$6"
  shift 6

  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"
  tmux new-session -d -s "$session" /bin/bash -lc "
    source '$CONDA_SH' &&
    conda activate '$ENV_NAME' &&
    cd '$REPO_ROOT' &&
    CUDA_VISIBLE_DEVICES='$gpu' python SDK/train_mcts.py \
      --resume-from '$RESUME_FROM' \
      --checkpoint-path '$checkpoint' \
      --champion-path '$champion' \
      --export-agent-model '$export_model' \
      --run-name '$run_name' \
      $*
  "
  printf 'launched session=%s gpu=%s run_name=%s\n' "$session" "$gpu" "$run_name"
}

launch_run \
  0 \
  "antgame-fast-${STAMP}" \
  "checkpoints/${STAMP}_fast_latest.npz" \
  "checkpoints/${STAMP}_fast_champion.npz" \
  "AI/${STAMP}_fast_ai_mcts_model.npz" \
  "${STAMP}-fast-lane" \
  --batches 12 \
  --episodes 8 \
  --search-iterations 32 \
  --max-depth 3 \
  --max-rounds 96 \
  --evaluation-episodes 2 \
  --opponent-pool-size 6 \
  --selfplay-mirror-ratio 0.20 \
  --selfplay-heuristic-ratio 0.35 \
  --promotion-episodes 4 \
  --promotion-win-rate 0.55

launch_run \
  1 \
  "antgame-pressure-${STAMP}" \
  "checkpoints/${STAMP}_pressure_latest.npz" \
  "checkpoints/${STAMP}_pressure_champion.npz" \
  "AI/${STAMP}_pressure_ai_mcts_model.npz" \
  "${STAMP}-pressure-line" \
  --batches 12 \
  --episodes 8 \
  --search-iterations 48 \
  --max-depth 4 \
  --max-rounds 128 \
  --evaluation-episodes 2 \
  --opponent-pool-size 8 \
  --selfplay-mirror-ratio 0.15 \
  --selfplay-heuristic-ratio 0.30 \
  --promotion-episodes 4 \
  --promotion-win-rate 0.56

launch_run \
  2 \
  "antgame-league-${STAMP}" \
  "checkpoints/${STAMP}_league_latest.npz" \
  "checkpoints/${STAMP}_league_champion.npz" \
  "AI/${STAMP}_league_ai_mcts_model.npz" \
  "${STAMP}-league-line" \
  --batches 10 \
  --episodes 10 \
  --search-iterations 40 \
  --max-depth 4 \
  --max-rounds 128 \
  --evaluation-episodes 2 \
  --opponent-pool-size 10 \
  --selfplay-mirror-ratio 0.10 \
  --selfplay-heuristic-ratio 0.25 \
  --promotion-episodes 6 \
  --promotion-win-rate 0.57

printf '\nresume_from=%s\nstamp=%s\n' "$RESUME_FROM" "$STAMP"
printf 'attach with: tmux attach -t <session-name>\n'
