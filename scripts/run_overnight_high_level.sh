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

check_oom() {
  # Check if our process or children were OOM-killed
  local pid="$1"
  local name="$2"

  # Check dmesg for OOM kill messages (needs permissions, may fail)
  if oom_line=$(dmesg 2>/dev/null | grep -i "out of memory\|oom-kill\|killed process" | tail -5); then
    if [ -n "$oom_line" ]; then
      printf '[%s] OOM-killer activity detected in dmesg:\n%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$oom_line"
    fi
  fi

  # Check /var/log/syslog or kern.log as fallback
  for syslog in /var/log/kern.log /var/log/syslog; do
    if [ -r "$syslog" ]; then
      if oom_syslog=$(grep -i "oom-kill\|out of memory\|killed process" "$syslog" 2>/dev/null | tail -3); then
        if [ -n "$oom_syslog" ]; then
          printf '[%s] OOM evidence in %s:\n%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$syslog" "$oom_syslog"
        fi
      fi
      break
    fi
  done
}

on_error() {
  local exit_code=$?
  printf '\n[%s] FAILED %s (exit=%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${CURRENT_PHASE:-unknown}" "$exit_code"

  # Decode well-known signal exit codes
  case "$exit_code" in
    137) printf '[%s] Exit 137 (SIGKILL) — likely OOM-killed by kernel\n' "$(date '+%Y-%m-%d %H:%M:%S')" ;;
    139) printf '[%s] Exit 139 (SIGSEGV) — segfault in native library (corrupted audio file?)\n' "$(date '+%Y-%m-%d %H:%M:%S')" ;;
    134) printf '[%s] Exit 134 (SIGABRT) — aborted, possibly assertion failure in native code\n' "$(date '+%Y-%m-%d %H:%M:%S')" ;;
  esac

  # Print memory state at time of failure
  printf '[%s] Memory at failure:\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  free -h 2>/dev/null || true

  check_oom "$$" "overnight_high_level"

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

CURRENT_PHASE="Configuration"
phase "$CURRENT_PHASE"
echo "FEATURE_WORKERS=$FEATURE_WORKERS"
echo "MAX_TASKS_PER_CHILD=$MAX_TASKS_PER_CHILD"
echo "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "FORMULA_SETS=$FORMULA_SETS"
echo "SEEDS=$SEEDS"
echo "FORMULA_OUTPUT_DIR=$FORMULA_OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
printf 'Memory: %s\n' "$(free -h 2>/dev/null | awk '/^Mem:/{print "total="$2" used="$3" avail="$7}' || echo 'unknown')"

CURRENT_PHASE="Sync environment"
phase "$CURRENT_PHASE"
uv sync

CURRENT_PHASE="Extract full audio features"
phase "$CURRENT_PHASE"
# ffmpeg on PATH enables a quick decode preflight so corrupt MP3s are skipped before librosa
# (reduces worker segfaults). Default multiprocessing is spawn (override: EXTRACT_MP_METHOD=fork).
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python src/extract_features.py \
  --manifest src/data/audio_manifest.csv \
  --output src/data/audio_features.csv \
  --feature-set full \
  --workers "$FEATURE_WORKERS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --max-tasks-per-child "$MAX_TASKS_PER_CHILD" \
  --failure-log-csv src/data/audio_features.failures.csv

CURRENT_PHASE="Optimize high-level formula sets"
phase "$CURRENT_PHASE"
uv run python src/optimize_high_level_formulas.py \
  --formula-sets "$FORMULA_SETS" \
  --seeds "$SEEDS" \
  --output-dir "$FORMULA_OUTPUT_DIR"

phase "Complete"
echo "Best formula result: $FORMULA_OUTPUT_DIR/best_formula_result.json"
echo "Full summary: $FORMULA_OUTPUT_DIR/formula_search_summary.csv"
echo "Run log: $LOG_FILE"
