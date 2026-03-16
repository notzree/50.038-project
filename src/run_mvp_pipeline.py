import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def run_step(step_name: str, command: list[str], cwd: Path) -> None:
    print(f"\n=== {step_name} ===")
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def build_manifest(songs_dir: Path, manifest_path: Path) -> int:
    print("\n=== Build manifest ===")
    print(f"Songs dir: {songs_dir}")
    print(f"Manifest: {manifest_path}")

    songs = sorted(songs_dir.glob("*.mp3"))
    rows = [{"track_id": p.stem, "file_path": str(p)} for p in songs]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pl.DataFrame(
            rows, schema={"track_id": pl.String, "file_path": pl.String}
        ).write_csv(manifest_path)
    else:
        pl.DataFrame(schema={"track_id": pl.String, "file_path": pl.String}).write_csv(
            manifest_path
        )

    print(f"Manifest rows: {len(rows)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end MVP pipeline")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run song download step before feature extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max songs to download when --download is used",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Download workers when --download is used",
    )
    parser.add_argument(
        "--songs-dir",
        default="src/data/songs",
        help="Directory containing downloaded .mp3 files",
    )
    parser.add_argument(
        "--manifest",
        default="src/data/audio_manifest.csv",
        help="Manifest CSV output path",
    )
    parser.add_argument(
        "--features-out",
        default="src/data/audio_features_basic.csv",
        help="Audio features CSV output path",
    )
    parser.add_argument(
        "--charts",
        default="src/data/charts.csv",
        help="Charts CSV path",
    )
    parser.add_argument(
        "--chart-name",
        default="top200",
        help="Chart name for labeling",
    )
    parser.add_argument(
        "--labels-out",
        default="src/data/labels_appears_in_region.csv",
        help="Labels CSV output path",
    )
    parser.add_argument(
        "--train-out",
        default="src/data/train_table_mvp.csv",
        help="MVP training table CSV output path",
    )
    parser.add_argument(
        "--metrics-out",
        default="src/data/mvp_metrics.json",
        help="Model metrics JSON output path",
    )
    parser.add_argument(
        "--model-out",
        default="src/data/mvp_model.joblib",
        help="Serialized selected model output path",
    )
    parser.add_argument(
        "--region-metrics-out",
        default="src/data/mvp_region_metrics.csv",
        help="Per-region metrics CSV output path",
    )
    parser.add_argument(
        "--test-preds-out",
        default="src/data/mvp_test_predictions.csv",
        help="Test predictions CSV output path",
    )
    parser.add_argument(
        "--errors-out",
        default="src/data/mvp_error_rows.csv",
        help="Error rows CSV output path",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip initial visualization generation",
    )
    parser.add_argument(
        "--viz-out",
        default="src/data/plots/mvp_label_distribution.png",
        help="Initial visualization PNG output path",
    )
    parser.add_argument(
        "--predict-audio",
        default=None,
        help="Optional audio path for single-song prediction after training",
    )
    parser.add_argument(
        "--predict-region",
        default=None,
        help="Optional region for single-song prediction after training",
    )
    parser.add_argument(
        "--predict-output-json",
        default=None,
        help="Optional JSON output path for single-song prediction",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    songs_dir = (repo_root / args.songs_dir).resolve()
    manifest_path = (repo_root / args.manifest).resolve()

    print("Starting MVP pipeline")
    print(f"Repo root: {repo_root}")

    if args.download:
        cmd = [
            sys.executable,
            "src/main.py",
            "--max-workers",
            str(args.max_workers),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        run_step("Download songs", cmd, cwd=repo_root)
    else:
        print("\n=== Download songs ===")
        print("Skipping download step (use --download to enable)")

    songs_dir.mkdir(parents=True, exist_ok=True)
    song_count = build_manifest(songs_dir=songs_dir, manifest_path=manifest_path)
    if song_count == 0:
        print("No songs found. Stopping pipeline.")
        print("Add songs to src/data/songs or rerun with --download")
        return

    run_step(
        "Extract features",
        [
            sys.executable,
            "src/extract_features.py",
            "--manifest",
            args.manifest,
            "--output",
            args.features_out,
        ],
        cwd=repo_root,
    )

    run_step(
        "Build labels and train table",
        [
            sys.executable,
            "src/build_mvp_dataset.py",
            "--charts",
            args.charts,
            "--features",
            args.features_out,
            "--labels-out",
            args.labels_out,
            "--train-out",
            args.train_out,
            "--chart-name",
            args.chart_name,
        ],
        cwd=repo_root,
    )

    run_step(
        "Train and evaluate models",
        [
            sys.executable,
            "src/train_mvp_model.py",
            "--input",
            args.train_out,
            "--metrics-out",
            args.metrics_out,
            "--model-out",
            args.model_out,
            "--region-metrics-out",
            args.region_metrics_out,
            "--test-preds-out",
            args.test_preds_out,
            "--errors-out",
            args.errors_out,
        ],
        cwd=repo_root,
    )

    if not args.skip_visualization:
        run_step(
            "Make initial visualization",
            [
                sys.executable,
                "src/make_initial_visualization.py",
                "--labels",
                args.labels_out,
                "--out",
                args.viz_out,
            ],
            cwd=repo_root,
        )
    else:
        print("\n=== Make initial visualization ===")
        print("Skipping visualization step (--skip-visualization enabled)")

    if args.predict_audio and args.predict_region:
        predict_cmd = [
            sys.executable,
            "src/predict_single_audio.py",
            "--audio",
            args.predict_audio,
            "--region",
            args.predict_region,
            "--model",
            args.model_out,
        ]
        if args.predict_output_json:
            predict_cmd.extend(["--output-json", args.predict_output_json])

        run_step("Predict single song", predict_cmd, cwd=repo_root)
    elif args.predict_audio or args.predict_region:
        print("\n=== Predict single song ===")
        print(
            "Skipping prediction: provide both --predict-audio and --predict-region "
            "to run single-song inference."
        )
    else:
        print("\n=== Predict single song ===")
        print("Skipping prediction step (no --predict-audio/--predict-region provided)")

    print("\nMVP pipeline complete")
    print(f"Features: {args.features_out}")
    print(f"Labels: {args.labels_out}")
    print(f"Train table: {args.train_out}")
    print(f"Metrics: {args.metrics_out}")
    print(f"Model: {args.model_out}")
    print(f"Region metrics: {args.region_metrics_out}")
    print(f"Test predictions: {args.test_preds_out}")
    print(f"Error rows: {args.errors_out}")
    if not args.skip_visualization:
        print(f"Visualization: {args.viz_out}")


if __name__ == "__main__":
    main()
