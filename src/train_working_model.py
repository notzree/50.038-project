"""Minimal pipeline: formula A → labels/train table → one logistic model (no CV, no search).

Use this when the full formula search / multi-model path is unstable or too slow.
Artifacts land under ``src/data/working_model/`` by default.
"""

import argparse
import os

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

from pathlib import Path

from build_dataset import build_labels_and_train_table
from optimize_human_formulas import build_high_level_features_for_set
from train_model import train_and_evaluate


def main() -> None:
    import faulthandler

    faulthandler.enable()

    parser = argparse.ArgumentParser(
        description="Build formula-A high-level features + train table, then train one LR model (no CV)"
    )
    parser.add_argument("--features-csv", default="src/data/audio_features.csv")
    parser.add_argument("--charts-csv", default="src/data/charts.csv")
    parser.add_argument("--track-catalog-csv", default="src/data/track_catalog.csv")
    parser.add_argument("--genres-csv", default=None)
    parser.add_argument(
        "--nonviral-meta-csv", default="src/data/nonviral_track_ids.csv"
    )
    parser.add_argument("--chart-name", default="top200")
    parser.add_argument(
        "--output-dir",
        default="src/data/working_model",
        help="Directory for high_level CSV, labels, train_table, and model artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip high-level + dataset build; expect train_table.parquet (or legacy .csv) under output-dir",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    high_level_csv = str(out / "audio_features_high_level.csv")
    labels_csv = str(out / "labels_appears_in_region.csv")
    train_parquet = out / "train_table.parquet"
    train_csv_legacy = out / "train_table.csv"
    train_csv = str(train_parquet)
    run_dir = out / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.train_only:
        if train_parquet.is_file():
            train_csv = str(train_parquet)
        elif train_csv_legacy.is_file():
            train_csv = str(train_csv_legacy)
        else:
            raise SystemExit(
                f"--train-only but neither {train_parquet} nor {train_csv_legacy} exists"
            )

    if not args.train_only:
        print("=== Step 1/3: high-level features (formula set A only) ===", flush=True)
        build_high_level_features_for_set(args.features_csv, high_level_csv, "A")
        print("=== Step 2/3: labels + train_table ===", flush=True)
        build_labels_and_train_table(
            charts_csv=args.charts_csv,
            features_csv=args.features_csv,
            track_catalog_csv=args.track_catalog_csv,
            high_level_features_csv=high_level_csv,
            labels_csv=labels_csv,
            train_csv=train_csv,
            chart_name=args.chart_name,
            genres_csv=args.genres_csv,
            nonviral_meta_csv=args.nonviral_meta_csv,
        )

    print("=== Step 3/3: train logistic regression (no CV) ===", flush=True)
    train_and_evaluate(
        input_csv=train_csv,
        metrics_out=str(run_dir / "model_metrics.json"),
        model_out=str(run_dir / "model.joblib"),
        region_metrics_out=str(run_dir / "region_metrics.csv"),
        test_preds_out=str(run_dir / "test_predictions.csv"),
        errors_out=str(run_dir / "error_rows.csv"),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        feature_importance_out=str(run_dir / "feature_importance.csv"),
        metadata_out=str(run_dir / "model_metadata.json"),
        cv_folds=2,
        rf_n_estimators=64,
        models=("logistic_regression",),
        skip_cross_validation=True,
    )
    print(
        f"\nDone. Model: {run_dir / 'model.joblib'}  metadata: {run_dir / 'model_metadata.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
