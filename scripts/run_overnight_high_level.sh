#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/overnight_high_level_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

phase() {
  printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

snapshot() {
  phase "System snapshot"
  uname -a || true
  if command -v free >/dev/null 2>&1; then
    free -h || true
  fi
  if command -v vm_stat >/dev/null 2>&1; then
    vm_stat || true
  fi
  df -h . || true
}

on_error() {
  local exit_code=$?
  printf '\n[%s] FAILED at line %s (exit=%s)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${BASH_LINENO[0]}" "$exit_code"
  printf '[%s] Log file: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$LOG_FILE"
  snapshot
  exit "$exit_code"
}

trap on_error ERR

FEATURE_WORKERS="${FEATURE_WORKERS:-2}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
FORMULA_SETS="${FORMULA_SETS:-A,B,C,D}"
SEEDS="${SEEDS:-42}"

phase "Run configuration"
echo "FEATURE_WORKERS=$FEATURE_WORKERS"
echo "MAX_TASKS_PER_CHILD=$MAX_TASKS_PER_CHILD"
echo "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "FORMULA_SETS=$FORMULA_SETS"
echo "SEEDS=$SEEDS"
echo "LOG_FILE=$LOG_FILE"

snapshot

phase "Sync environment"
uv sync

phase "Extract full audio features"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python src/extract_features.py \
  --manifest src/data/audio_manifest.csv \
  --output src/data/audio_features.csv \
  --feature-set full \
  --workers "$FEATURE_WORKERS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --max-tasks-per-child "$MAX_TASKS_PER_CHILD" \
  --failure-log-csv src/data/audio_features.failures.csv

phase "Build high-level features"
uv run python src/engineer_high_level_features.py \
  --input src/data/audio_features.csv \
  --output src/data/audio_features_high_level.csv

phase "Build labels and training table"
uv run python src/build_dataset.py \
  --features src/data/audio_features.csv \
  --high-level-features src/data/audio_features_high_level.csv

phase "Train model"
uv run python src/train_model.py \
  --input src/data/train_table.csv \
  --metrics-out src/data/model_metrics.json \
  --model-out src/data/model.joblib \
  --region-metrics-out src/data/region_metrics.csv

phase "Optimize high-level formula sets"
uv run python src/optimize_high_level_formulas.py \
  --formula-sets "$FORMULA_SETS" \
  --seeds "$SEEDS" \
  --output-dir src/data/formula_search

phase "Complete"
echo "Overnight run completed successfully"
echo "Log file: $LOG_FILE"
