#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/overnight_high_level_${STAMP}.log"
LOCK_FILE="$LOG_DIR/overnight_high_level.pid"

exec > >(tee -a "$LOG_FILE") 2>&1

phase() {
  printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cleanup() {
  if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$LOCK_FILE" || true
  fi
}

on_error() {
  local exit_code=$?
  printf '\n[%s] FAILED (exit=%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$exit_code"
  printf '[%s] Log file: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$LOG_FILE"
  exit "$exit_code"
}

trap on_error ERR
trap cleanup EXIT

if [ -f "$LOCK_FILE" ]; then
  old_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "Another run is active (pid=$old_pid)."
    echo "If stale, remove lock: rm -f $LOCK_FILE"
    exit 1
  fi
fi
echo "$$" > "$LOCK_FILE"

FEATURE_WORKERS="${FEATURE_WORKERS:-1}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
FORMULA_SETS="${FORMULA_SETS:-A,B,C,D}"
SEEDS="${SEEDS:-42}"
FORMULA_OUTPUT_DIR="${FORMULA_OUTPUT_DIR:-src/data/formula_search}"

phase "Configuration"
echo "FEATURE_WORKERS=$FEATURE_WORKERS"
echo "MAX_TASKS_PER_CHILD=$MAX_TASKS_PER_CHILD"
echo "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "FORMULA_SETS=$FORMULA_SETS"
echo "SEEDS=$SEEDS"
echo "FORMULA_OUTPUT_DIR=$FORMULA_OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"

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

phase "Optimize high-level formula sets"
uv run python src/optimize_high_level_formulas.py \
  --formula-sets "$FORMULA_SETS" \
  --seeds "$SEEDS" \
  --output-dir "$FORMULA_OUTPUT_DIR"

phase "Complete"
echo "Best formula result: $FORMULA_OUTPUT_DIR/best_formula_result.json"
echo "Full summary: $FORMULA_OUTPUT_DIR/formula_search_summary.csv"
echo "Run log: $LOG_FILE"
