import argparse
import csv
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Suppress noisy librosa warnings (PySoundFile fallback, audioread deprecation)
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050

# Normalization constants for engineered scores (empirical, tunable)
MAX_TEMPO_BPM = 180.0
MAX_ONSET_STRENGTH = 20.0
MAX_TEMPO_INSTABILITY = 0.5
MAX_RMS = 0.2
MAX_SPECTRAL_CENTROID = 5000.0
MAX_SPECTRAL_BANDWIDTH = 3000.0
BASS_CUTOFF_HZ = 300

# ---------------------------------------------------------------------------
# Feature key lists
# ---------------------------------------------------------------------------

BASIC_FEATURE_KEYS = [
    "duration_sec_extracted",
    "sample_rate_extracted",
    "tempo_bpm",
    "rms_mean",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "zcr_mean",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
]

FULL_FEATURE_KEYS = [
    # Basic (kept from original)
    "duration_sec_extracted",
    "sample_rate_extracted",
    "tempo_bpm",
    "rms_mean",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "zcr_mean",
    # MFCCs 1-13 mean + std
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
    *[f"mfcc_{i}_std" for i in range(1, 14)],
    # Spectral bandwidth
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    # Spectral contrast (7 bands)
    *[f"spectral_contrast_{i}_mean" for i in range(7)],
    # Chroma STFT (12 pitch classes)
    *[f"chroma_{i}_mean" for i in range(12)],
    # Temporal dynamics (std of frame-level features)
    "rms_std",
    "spectral_centroid_std",
    "spectral_rolloff_std",
    "zcr_std",
    # Rhythm & beat
    "onset_strength_mean",
    "onset_strength_std",
    "beat_strength_mean",
    "tempo_stability",
    # Engineered scores
    "danceability_score",
    "energy_score",
    # Energy trajectory
    "energy_arc_slope",
    "energy_arc_std",
    "spectral_flux_mean",
    "spectral_flux_std",
    # Self-similarity
    "chroma_self_similarity_mean",
    # Event density
    "onset_density",
    "harmonic_change_rate",
    # Timbral texture
    "bass_energy_ratio",
    "brightness_slope",
]


# ---------------------------------------------------------------------------
# Core audio loading + shared frame-level computation
# ---------------------------------------------------------------------------


def _load_and_compute_core(audio_path: str) -> dict:
    """Load audio and compute core frame-level features shared by both basic and full.

    Returns a dict with raw arrays (for further computation) and scalar features.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if y.size == 0:
        raise ValueError("empty audio")

    duration_sec = float(len(y) / sr)

    # Use librosa.feature.tempo instead of beat_track (beat_track segfaults on macOS ARM)
    tempo_arr = librosa.feature.tempo(y=y, sr=sr)
    tempo_bpm = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0

    # Get beat frames via onset envelope + peak picking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
    )

    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "y": y,
        "sr": sr,
        "duration_sec": duration_sec,
        "tempo_bpm": tempo_bpm,
        "beat_frames": beat_frames,
        "onset_env": onset_env,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "zcr": zcr,
    }


# ---------------------------------------------------------------------------
# Basic feature extraction
# ---------------------------------------------------------------------------


def extract_basic_features(audio_path: str) -> dict:
    core = _load_and_compute_core(audio_path)
    mfcc = librosa.feature.mfcc(y=core["y"], sr=core["sr"], n_mfcc=3)

    return {
        "duration_sec_extracted": core["duration_sec"],
        "sample_rate_extracted": int(core["sr"]),
        "tempo_bpm": core["tempo_bpm"],
        "rms_mean": float(np.mean(core["rms"])),
        "spectral_centroid_mean": float(np.mean(core["spectral_centroid"])),
        "spectral_rolloff_mean": float(np.mean(core["spectral_rolloff"])),
        "zcr_mean": float(np.mean(core["zcr"])),
        "mfcc_1_mean": float(np.mean(mfcc[0])),
        "mfcc_2_mean": float(np.mean(mfcc[1])),
        "mfcc_3_mean": float(np.mean(mfcc[2])),
    }


# ---------------------------------------------------------------------------
# Full feature extraction (~46 features, optimized for 30s clips)
# ---------------------------------------------------------------------------


def extract_full_features(audio_path: str) -> dict:
    core = _load_and_compute_core(audio_path)
    y, sr = core["y"], core["sr"]
    rms = core["rms"]
    spectral_centroid = core["spectral_centroid"]
    spectral_rolloff = core["spectral_rolloff"]
    zcr = core["zcr"]
    beat_frames = core["beat_frames"]

    feats = {}

    # --- Basic ---
    feats["duration_sec_extracted"] = core["duration_sec"]
    feats["sample_rate_extracted"] = int(sr)
    feats["tempo_bpm"] = core["tempo_bpm"]
    feats["rms_mean"] = float(np.mean(rms))
    feats["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    feats["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    feats["zcr_mean"] = float(np.mean(zcr))

    # Temporal dynamics (std of frame-level)
    feats["rms_std"] = float(np.std(rms))
    feats["spectral_centroid_std"] = float(np.std(spectral_centroid))
    feats["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    feats["zcr_std"] = float(np.std(zcr))

    # --- MFCCs 1-13 mean + std ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    # --- Spectral bandwidth ---
    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["spectral_bandwidth_mean"] = float(np.mean(spectral_bw))
    feats["spectral_bandwidth_std"] = float(np.std(spectral_bw))

    # --- Spectral contrast (7 bands) ---
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spectral_contrast.shape[0]):
        feats[f"spectral_contrast_{i}_mean"] = float(np.mean(spectral_contrast[i]))

    # --- Chroma STFT (12 pitch classes) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))

    # --- Rhythm & beat features ---
    onset_env = core["onset_env"]
    feats["onset_strength_mean"] = float(np.mean(onset_env))
    feats["onset_strength_std"] = float(np.std(onset_env))

    if len(beat_frames) > 0:
        beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]
        feats["beat_strength_mean"] = (
            float(np.mean(beat_strengths)) if len(beat_strengths) > 0 else 0.0
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) > 1:
            feats["tempo_stability"] = float(np.std(np.diff(beat_times)))
        else:
            feats["tempo_stability"] = 0.0
    else:
        feats["beat_strength_mean"] = 0.0
        feats["tempo_stability"] = 0.0

    # --- Engineered scores ---
    tempo_norm = np.clip(feats["tempo_bpm"] / MAX_TEMPO_BPM, 0, 1)
    onset_norm = np.clip(feats["onset_strength_mean"] / MAX_ONSET_STRENGTH, 0, 1)
    stability_norm = 1.0 - np.clip(
        feats["tempo_stability"] / MAX_TEMPO_INSTABILITY, 0, 1
    )
    feats["danceability_score"] = float(
        (tempo_norm + onset_norm + stability_norm) / 3.0
    )

    rms_norm = np.clip(feats["rms_mean"] / MAX_RMS, 0, 1)
    centroid_norm = np.clip(
        feats["spectral_centroid_mean"] / MAX_SPECTRAL_CENTROID, 0, 1
    )
    bw_norm = np.clip(feats["spectral_bandwidth_mean"] / MAX_SPECTRAL_BANDWIDTH, 0, 1)
    feats["energy_score"] = float((rms_norm + centroid_norm + bw_norm) / 3.0)

    # --- Energy trajectory ---
    if len(rms) > 1:
        x = np.arange(len(rms))
        coeffs = np.polyfit(x, rms, 1)
        feats["energy_arc_slope"] = float(coeffs[0])
        residuals = rms - np.polyval(coeffs, x)
        feats["energy_arc_std"] = float(np.std(residuals))
    else:
        feats["energy_arc_slope"] = 0.0
        feats["energy_arc_std"] = 0.0

    S = np.abs(librosa.stft(y))
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    feats["spectral_flux_mean"] = (
        float(np.mean(spectral_flux)) if len(spectral_flux) > 0 else 0.0
    )
    feats["spectral_flux_std"] = (
        float(np.std(spectral_flux)) if len(spectral_flux) > 0 else 0.0
    )

    # --- Intra-clip self-similarity ---
    # Use running mean instead of full NxN similarity matrix to avoid OOM on long tracks
    chroma_norm = librosa.util.normalize(chroma, axis=0)
    n_frames = chroma_norm.shape[1]
    if n_frames > 1:
        # Sample pairs instead of building full O(n^2) matrix
        max_pairs = 5000
        if n_frames * (n_frames - 1) // 2 <= max_pairs:
            sim_matrix = chroma_norm.T @ chroma_norm
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            feats["chroma_self_similarity_mean"] = float(np.mean(upper_tri))
        else:
            rng = np.random.RandomState(42)
            idx_a = rng.randint(0, n_frames, size=max_pairs)
            idx_b = rng.randint(0, n_frames, size=max_pairs)
            mask = idx_a != idx_b
            idx_a, idx_b = idx_a[mask], idx_b[mask]
            dots = np.sum(chroma_norm[:, idx_a] * chroma_norm[:, idx_b], axis=0)
            feats["chroma_self_similarity_mean"] = float(np.mean(dots))
    else:
        feats["chroma_self_similarity_mean"] = 0.0

    # --- Event density ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = feats["duration_sec_extracted"]
    feats["onset_density"] = float(len(onsets) / duration) if duration > 0 else 0.0

    chroma_diff = np.diff(chroma, axis=1)
    chroma_change = np.sqrt(np.sum(chroma_diff**2, axis=0))
    feats["harmonic_change_rate"] = (
        float(np.mean(chroma_change)) if len(chroma_change) > 0 else 0.0
    )

    # --- Timbral texture ---
    # Free raw audio — no longer needed
    del y
    S_power = S**2
    freq_bins = librosa.fft_frequencies(sr=sr)
    bass_mask = freq_bins <= BASS_CUTOFF_HZ
    total_energy = np.sum(S_power)
    bass_energy = np.sum(S_power[bass_mask, :])
    feats["bass_energy_ratio"] = float(bass_energy / (total_energy + 1e-10))
    del S, S_power

    if len(spectral_centroid) > 1:
        x = np.arange(len(spectral_centroid))
        coeffs = np.polyfit(x, spectral_centroid, 1)
        feats["brightness_slope"] = float(coeffs[0])
    else:
        feats["brightness_slope"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# Worker function for parallel extraction
# ---------------------------------------------------------------------------


def _extract_one(args: tuple) -> dict:
    """Worker function — runs extraction and catches Python exceptions.

    NOTE: This cannot catch segfaults (SIGSEGV/exit 139) in native libraries.
    Use _extract_one_isolated() to run this in a subprocess for crash safety.
    """
    track_id, file_path, feature_set = args
    extract_fn = (
        extract_full_features if feature_set == "full" else extract_basic_features
    )
    feature_keys = FULL_FEATURE_KEYS if feature_set == "full" else BASIC_FEATURE_KEYS

    try:
        feats = extract_fn(file_path)
        return {
            "track_id": track_id,
            "file_path": file_path,
            "status": "ok",
            "error": None,
            **feats,
        }
    except Exception as e:
        err = str(e).strip()
        if not err:
            err = repr(e)
        return {
            "track_id": track_id,
            "file_path": file_path,
            "status": "failed",
            "error": f"{type(e).__name__}: {err}",
            **{k: None for k in feature_keys},
        }




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_extraction(
    manifest_csv: str,
    output_csv: str,
    feature_set: str = "full",
    workers: int | None = None,
    checkpoint_interval: int = 1000,
    max_tasks_per_child: int = 100,
    failure_log_csv: str | None = None,
    progress_interval: int = 10,
) -> None:
    """Extract features from all tracks in a manifest CSV."""
    manifest_path = Path(manifest_csv)
    output_path = Path(output_csv)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"Starting feature extraction (feature_set={feature_set}, workers={workers})")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Max tasks per child: {max_tasks_per_child}")
    print(f"Progress interval: {progress_interval}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = {"track_id", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    # Skip already-extracted tracks if output exists (resume support)
    already_done = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            already_done = set(existing.loc[existing["status"] == "ok", "track_id"])
            print(f"Resuming: {len(already_done)} tracks already extracted")
        except Exception:
            pass

    rows = [
        row
        for _, row in df[["track_id", "file_path"]].iterrows()
        if row["track_id"] not in already_done
    ]

    if not rows:
        print("All tracks already extracted, nothing to do")
        return

    print(f"Extracting features for {len(rows)} tracks...")

    tasks = [(row["track_id"], row["file_path"], feature_set) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if failure_log_csv is None:
        failure_log_path = output_path.with_suffix(".failures.csv")
    else:
        failure_log_path = Path(failure_log_csv)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    existing_failure_log = (
        failure_log_path.exists() and failure_log_path.stat().st_size > 0
    )
    failure_file = failure_log_path.open("a", newline="")
    failure_writer = csv.writer(failure_file)
    if not existing_failure_log:
        failure_writer.writerow(["track_id", "file_path", "error"])
        failure_file.flush()

    print(f"Failure log: {failure_log_path}")

    # Determine CSV header and open output for append-mode writing
    feature_keys = FULL_FEATURE_KEYS if feature_set == "full" else BASIC_FEATURE_KEYS
    fieldnames = ["track_id", "file_path", "status", "error"] + list(feature_keys)

    output_file_exists = output_path.exists() and output_path.stat().st_size > 0
    output_file = output_path.open("a", newline="")
    output_writer = csv.DictWriter(
        output_file, fieldnames=fieldnames, extrasaction="ignore"
    )
    if not output_file_exists:
        output_writer.writeheader()
        output_file.flush()

    completed = 0
    ok_count = 0
    failed_count = 0
    start_ts = time.time()

    def _handle_result(result: dict) -> None:
        nonlocal completed, ok_count, failed_count
        completed += 1

        # Write result row immediately instead of accumulating in memory
        output_writer.writerow(result)

        if result["status"] == "failed":
            failed_count += 1
            print(
                f"  [{completed}/{len(tasks)}] FAILED {result['track_id']}: {result['error']}"
            )
            failure_writer.writerow(
                [result["track_id"], result["file_path"], result["error"]]
            )
            failure_file.flush()
        else:
            ok_count += 1

        should_checkpoint = completed % checkpoint_interval == 0
        if should_checkpoint:
            output_file.flush()

        if completed % progress_interval == 0 or completed == len(tasks):
            elapsed = max(time.time() - start_ts, 1e-6)
            rate = completed / elapsed
            pct = completed * 100 // len(tasks)
            checkpoint_tag = " [checkpointed]" if should_checkpoint else ""
            print(
                f"  [{completed}/{len(tasks)} {pct}%] ok={ok_count} failed={failed_count} rate={rate:.1f}/s{checkpoint_tag}",
                flush=True,
            )

    # Always use a pool (even workers=1) so segfaults kill a worker, not the parent.
    # On BrokenProcessPool (worker segfault), log the crashing track, skip it,
    # and restart the pool for remaining tasks.
    import sys

    mp_method = "spawn" if sys.platform == "darwin" else "fork"
    ctx = multiprocessing.get_context(mp_method)
    # max_tasks_per_child is only supported with 'spawn'; drop it for 'fork'
    pool_kwargs = dict(max_workers=workers, mp_context=ctx)
    if mp_method == "spawn":
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child

    try:
        remaining = list(tasks)
        while remaining:
            try:
                with ProcessPoolExecutor(**pool_kwargs) as pool:
                    for result in pool.map(_extract_one, remaining, chunksize=1):
                        _handle_result(result)
                remaining = []
            except BrokenProcessPool:
                output_file.flush()
                # completed tracks have been handled; the next one crashed the worker
                crashed_task = tasks[completed]
                print(
                    f"  [{completed + 1}/{len(tasks)}] CRASHED {crashed_task[0]}: "
                    f"worker segfault — skipping",
                    flush=True,
                )
                _handle_result(
                    {
                        "track_id": crashed_task[0],
                        "file_path": crashed_task[1],
                        "status": "failed",
                        "error": "Worker crashed (segfault)",
                        **{k: None for k in feature_keys},
                    }
                )
                remaining = tasks[completed:]
                if remaining:
                    print(
                        f"  Restarting pool for {len(remaining)} remaining tracks...",
                        flush=True,
                    )
    finally:
        output_file.close()
        failure_file.close()

    print(
        f"Wrote {output_path} | ok={ok_count + len(already_done)}, failed={failed_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features from manifest."
    )
    parser.add_argument(
        "--manifest",
        default="src/data/audio_manifest.csv",
        help="Path to manifest CSV with columns: track_id, file_path",
    )
    parser.add_argument(
        "--output",
        default="src/data/audio_features.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--feature-set",
        choices=["basic", "full"],
        default="full",
        help="Feature set to extract (default: full)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Write checkpoint every N songs",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Restart each worker after N tracks to reduce long-run memory pressure",
    )
    parser.add_argument(
        "--failure-log-csv",
        default=None,
        help="Optional CSV path for per-track failures",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress summary every N processed tracks",
    )
    args = parser.parse_args()

    run_extraction(
        manifest_csv=args.manifest,
        output_csv=args.output,
        feature_set=args.feature_set,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        max_tasks_per_child=args.max_tasks_per_child,
        failure_log_csv=args.failure_log_csv,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
