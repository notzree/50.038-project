import argparse
import json
from pathlib import Path

import pandas as pd


def run_qc(
    manifest_csv: str,
    features_csv: str | None,
    summary_out: str,
    issues_out: str,
) -> dict:
    print("Running audio dataset QC")
    print(f"Manifest: {manifest_csv}")
    if features_csv:
        print(f"Features: {features_csv}")

    manifest = pd.read_csv(manifest_csv)
    required = {"track_id", "file_path"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {missing_cols}")

    manifest = manifest.copy()
    manifest["file_exists"] = manifest["file_path"].apply(
        lambda p: Path(str(p)).exists()
    )

    dup_track_count = int(manifest["track_id"].duplicated().sum())
    dup_path_count = int(manifest["file_path"].duplicated().sum())
    missing_file_count = int((~manifest["file_exists"]).sum())

    feature_summary = {}
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv)
        if "status" in feats.columns:
            n_ok = int((feats["status"] == "ok").sum())
            n_failed = int((feats["status"] == "failed").sum())
            extraction_success_rate = float(n_ok / max(n_ok + n_failed, 1))
        else:
            n_ok = int(len(feats))
            n_failed = 0
            extraction_success_rate = 1.0

        feature_summary = {
            "feature_rows": int(len(feats)),
            "feature_ok_rows": n_ok,
            "feature_failed_rows": n_failed,
            "feature_success_rate": extraction_success_rate,
        }

    issues = manifest[
        (~manifest["file_exists"]) | manifest["track_id"].duplicated(keep=False)
    ]
    issues_path = Path(issues_out)
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(issues_path, index=False)

    summary = {
        "manifest_rows": int(len(manifest)),
        "unique_tracks": int(manifest["track_id"].nunique()),
        "duplicate_track_id_rows": dup_track_count,
        "duplicate_file_path_rows": dup_path_count,
        "missing_file_rows": missing_file_count,
        "manifest_file_exists_rate": float((manifest["file_exists"]).mean()),
        **feature_summary,
    }

    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote QC summary to {summary_out}")
    print(f"Wrote QC issues to {issues_out}")
    print(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manifest/features QC checks")
    parser.add_argument("--manifest", default="src/data/audio_manifest.csv")
    parser.add_argument(
        "--features",
        default="src/data/audio_features.csv",
        help="Optional features CSV path",
    )
    parser.add_argument(
        "--summary-out",
        default="src/data/audio_qc_summary.json",
        help="Output JSON path for QC summary",
    )
    parser.add_argument(
        "--issues-out",
        default="src/data/audio_qc_issues.csv",
        help="Output CSV path for QC issue rows",
    )
    args = parser.parse_args()

    run_qc(
        manifest_csv=args.manifest,
        features_csv=args.features,
        summary_out=args.summary_out,
        issues_out=args.issues_out,
    )


if __name__ == "__main__":
    main()
