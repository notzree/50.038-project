#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/overnight_high_level_${STAMP}.log"
MONITOR_DIR="$LOG_DIR/monitor_${STAMP}"
mkdir -p "$MONITOR_DIR"

MONITOR_PIDS=()

exec > >(tee -a "$LOG_FILE") 2>&1

phase() {
  printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

log_cmd() {
  local title="$1"
  shift
  echo "--- $title ---"
  "$@" || true
}

start_monitor() {
  local name="$1"
  shift
  local out="$MONITOR_DIR/${name}.log"
  echo "Starting monitor: $name -> $out"
  "$@" >"$out" 2>&1 &
  local pid=$!
  MONITOR_PIDS+=("$pid")
  echo "Monitor $name pid=$pid"
}

stop_monitors() {
  if [ "${#MONITOR_PIDS[@]}" -eq 0 ]; then
    return
  fi
  phase "Stopping monitors"
  for pid in "${MONITOR_PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      echo "Stopped monitor pid=$pid"
    fi
  done
}

collect_kernel_history() {
  phase "Kernel history"
  if have_cmd journalctl; then
    log_cmd "journalctl previous boot kernel" sh -c "journalctl -k -b -1 --no-pager | tail -n 300"
    log_cmd "journalctl current boot kernel" sh -c "journalctl -k -b 0 --no-pager | tail -n 300"
  fi
  if have_cmd dmesg; then
    log_cmd "dmesg tail" sh -c "dmesg -T | tail -n 300"
  fi
}

start_os_monitors() {
  phase "Starting OS monitors"
  echo "Monitor logs directory: $MONITOR_DIR"

  if have_cmd dmesg; then
    start_monitor "dmesg_watch" sh -c "dmesg -wT"
  fi
  if have_cmd vmstat; then
    start_monitor "vmstat" vmstat 5
  fi
  if have_cmd iostat; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      start_monitor "iostat" iostat -w 5
    else
      start_monitor "iostat" iostat -xz 5
    fi
  fi
  if have_cmd nvidia-smi; then
    start_monitor "nvidia_smi" nvidia-smi -l 5
  fi
  if have_cmd sensors; then
    start_monitor "sensors" sh -c "while true; do date; sensors; sleep 10; done"
  fi
  if have_cmd top; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      start_monitor "top" sh -c "while true; do top -l 1 -n 0; sleep 10; done"
    else
      start_monitor "top" top -b -d 10
    fi
  fi
}

snapshot() {
  phase "System snapshot"
  log_cmd "date" date
  log_cmd "uname" uname -a

  if have_cmd sw_vers; then
    log_cmd "sw_vers" sw_vers
  fi
  if [ -f /etc/os-release ]; then
    log_cmd "os-release" cat /etc/os-release
  fi
  if have_cmd lsb_release; then
    log_cmd "lsb_release" lsb_release -a
  fi

  if have_cmd free; then
    log_cmd "memory free -h" free -h
  fi
  if have_cmd vm_stat; then
    log_cmd "vm_stat" vm_stat
  fi
  if have_cmd top; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      log_cmd "top (Darwin)" top -l 1 -n 0
    else
      log_cmd "top (Linux)" top -b -n 1
    fi
  fi

  if [ -f /proc/cpuinfo ]; then
    log_cmd "cpuinfo" cat /proc/cpuinfo
  fi
  if have_cmd sysctl; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      log_cmd "sysctl hw.ncpu" sysctl hw.ncpu
      log_cmd "sysctl hw.memsize" sysctl hw.memsize
      log_cmd "sysctl machdep.cpu.brand_string" sysctl machdep.cpu.brand_string
    else
      log_cmd "sysctl kernel.ostype" sysctl kernel.ostype
      log_cmd "sysctl kernel.osrelease" sysctl kernel.osrelease
    fi
  fi
  if have_cmd nproc; then
    log_cmd "nproc" nproc
  fi

  log_cmd "ulimit" sh -c 'ulimit -a'
  log_cmd "disk usage" df -h .
}

environment_diagnostics() {
  phase "Environment diagnostics"

  log_cmd "pwd" pwd
  log_cmd "git branch" git status -sb

  if have_cmd uv; then
    log_cmd "uv version" uv --version
  fi
  if have_cmd python3; then
    log_cmd "python3 version" python3 --version
  fi

  if have_cmd ffmpeg; then
    log_cmd "ffmpeg version" ffmpeg -version
  fi
  if have_cmd ffprobe; then
    log_cmd "ffprobe version" ffprobe -version
  fi
  if have_cmd sox; then
    log_cmd "sox version" sox --version
  fi

  echo "--- threading env vars ---"
  env | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR|UV|PYTHON)' | sort || true

  echo "--- python package diagnostics (uv environment) ---"
  uv run python - <<'PY' || true
import importlib
import platform
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("system:", platform.system())
print("release:", platform.release())
print("machine:", platform.machine())
print("processor:", platform.processor())

packages = [
    "librosa",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "soundfile",
    "audioread",
]
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}_version:", version)
    except Exception as exc:
        print(f"{name}_version: IMPORT_FAILED ({type(exc).__name__}: {exc})")

try:
    import soundfile as sf
    print("soundfile_info:", sf.__libsndfile_version__)
except Exception as exc:
    print(f"soundfile_info: UNAVAILABLE ({type(exc).__name__}: {exc})")
PY

  echo "--- sample of manifest paths (first 5) ---"
  uv run python - <<'PY' || true
import pandas as pd

try:
    df = pd.read_csv("src/data/audio_manifest.csv", usecols=["track_id", "file_path"])
    print("manifest_rows:", len(df))
    print(df.head(5).to_string(index=False))
except Exception as exc:
    print("manifest_read_failed:", type(exc).__name__, exc)
PY
}

on_error() {
  local exit_code=$?
  printf '\n[%s] FAILED at line %s (exit=%s)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${BASH_LINENO[0]}" "$exit_code"
  printf '[%s] Log file: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$LOG_FILE"
  printf '[%s] Monitor logs: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$MONITOR_DIR"
  snapshot
  collect_kernel_history
  stop_monitors
  exit "$exit_code"
}

on_exit() {
  stop_monitors
}

trap on_error ERR
trap on_exit EXIT

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
echo "MONITOR_DIR=$MONITOR_DIR"

snapshot
environment_diagnostics
collect_kernel_history
start_os_monitors

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
echo "Monitor logs: $MONITOR_DIR"
