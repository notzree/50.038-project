import argparse
from pathlib import Path

import librosa
import numpy as np
import polars as pl


OUTPUT_SCHEMA = {
    "track_id": pl.String,
    "file_path": pl.String,
    "status": pl.String,
    "error": pl.String,
    "duration_sec_extracted": pl.Float64,
    "sample_rate_extracted": pl.Int64,
    "tempo_bpm": pl.Float64,
    "rms_mean": pl.Float64,
    "spectral_centroid_mean": pl.Float64,
    "spectral_rolloff_mean": pl.Float64,
    "zcr_mean": pl.Float64,
    "mfcc_1_mean": pl.Float64,
    "mfcc_2_mean": pl.Float64,
    "mfcc_3_mean": pl.Float64,
}


def extract_basic_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if y.size == 0:
        raise ValueError("empty audio")

    duration_sec = float(len(y) / sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)

    return {
        "duration_sec_extracted": duration_sec,
        "sample_rate_extracted": int(sr),
        "tempo_bpm": float(tempo),
        "rms_mean": float(np.mean(rms)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "mfcc_1_mean": float(np.mean(mfcc[0])),
        "mfcc_2_mean": float(np.mean(mfcc[1])),
        "mfcc_3_mean": float(np.mean(mfcc[2])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract basic audio features from manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="src/data/audio_manifest.csv",
        help="Path to manifest CSV/parquet with columns: track_id, file_path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/data/audio_features_basic.csv",
        help="Path to output CSV/parquet",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".csv":
        df = pl.read_csv(manifest_path)
    elif manifest_path.suffix.lower() == ".parquet":
        df = pl.read_parquet(manifest_path)
    else:
        raise ValueError("Manifest must be .csv or .parquet")

    required_cols = {"track_id", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    records = []
    for row in df.select(["track_id", "file_path"]).iter_rows(named=True):
        track_id = row["track_id"]
        file_path = row["file_path"]

        try:
            feats = extract_basic_features(file_path)
            records.append(
                {
                    "track_id": track_id,
                    "file_path": file_path,
                    "status": "ok",
                    "error": None,
                    **feats,
                }
            )
        except Exception as e:
            records.append(
                {
                    "track_id": track_id,
                    "file_path": file_path,
                    "status": "failed",
                    "error": str(e),
                    "duration_sec_extracted": None,
                    "sample_rate_extracted": None,
                    "tempo_bpm": None,
                    "rms_mean": None,
                    "spectral_centroid_mean": None,
                    "spectral_rolloff_mean": None,
                    "zcr_mean": None,
                    "mfcc_1_mean": None,
                    "mfcc_2_mean": None,
                    "mfcc_3_mean": None,
                }
            )

    if records:
        out_df = pl.DataFrame(records)
    else:
        out_df = pl.DataFrame(schema=OUTPUT_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        out_df.write_csv(output_path)
    elif output_path.suffix.lower() == ".parquet":
        out_df.write_parquet(output_path)
    else:
        raise ValueError("Output path must end in .csv or .parquet")

    n_ok = out_df.filter(pl.col("status") == "ok").height
    n_failed = out_df.filter(pl.col("status") == "failed").height
    print(f"Wrote {output_path} | ok={n_ok}, failed={n_failed}")


if __name__ == "__main__":
    main()
