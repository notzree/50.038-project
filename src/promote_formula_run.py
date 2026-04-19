from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def _snapshot_baseline(data_dir: Path) -> None:
    baseline_pairs = [
        (data_dir / "model_metrics.json", data_dir / "model_metrics_baseline.json"),
        (
            data_dir / "error_counts_by_region.csv",
            data_dir / "error_counts_by_region_baseline.csv",
        ),
    ]
    for src, dst in baseline_pairs:
        if src.exists():
            _copy(src, dst)


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = src.stat().st_size / (1024 * 1024)
    print(f"Copied ({size_mb:.1f} MB): {src} -> {dst}")


def _status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def _rebuild_with_country_profile(
    data_dir: Path,
    charts_csv: Path,
    features_csv: Path,
    track_catalog_csv: Path,
    high_level_features_csv: Path,
    genres_csv: str | None,
    nonviral_meta_csv: str | None,
    country_profile_csv: Path,
    chart_name: str,
    seed: int,
    cv_folds: int,
    cv_jobs: int,
    rf_jobs: int,
) -> None:
    from build_dataset import build_labels_and_train_table
    from train_model import train_and_evaluate

    print("\n=== Rebuild dataset with country profile features ===")
    build_labels_and_train_table(
        charts_csv=str(charts_csv),
        features_csv=str(features_csv),
        track_catalog_csv=str(track_catalog_csv),
        high_level_features_csv=str(high_level_features_csv),
        labels_csv=str(data_dir / "labels_appears_in_region.csv"),
        train_csv=str(data_dir / "train_table.csv"),
        chart_name=chart_name,
        genres_csv=genres_csv,
        nonviral_meta_csv=nonviral_meta_csv,
        country_profile_csv=str(country_profile_csv),
    )

    print("=== Retrain promoted model with country profile features ===")
    train_and_evaluate(
        input_csv=str(data_dir / "train_table.csv"),
        metrics_out=str(data_dir / "model_metrics.json"),
        model_out=str(data_dir / "model.joblib"),
        region_metrics_out=str(data_dir / "region_metrics.csv"),
        test_preds_out=str(data_dir / "test_predictions.csv"),
        errors_out=str(data_dir / "error_rows.csv"),
        test_size=0.2,
        val_size=0.2,
        seed=seed,
        feature_importance_out=str(data_dir / "feature_importance.csv"),
        metadata_out=str(data_dir / "model_metadata.json"),
        cv_folds=cv_folds,
        cv_jobs=cv_jobs,
        rf_n_jobs=rf_jobs,
    )


def promote(
    set_id: str,
    seed: int,
    base_dir: Path,
    data_dir: Path,
    charts_csv: Path,
    features_csv: Path,
    track_catalog_csv: Path,
    genres_csv: str | None,
    nonviral_meta_csv: str | None,
    chart_name: str,
    apply_country_profile: bool,
    country_profile_csv: Path,
    cv_folds: int,
    cv_jobs: int,
    rf_jobs: int,
) -> None:
    started = time.perf_counter()
    set_dir = base_dir / f"set_{set_id}"
    seed_dir = set_dir / f"seed_{seed}"

    print("\n=== Promotion config ===")
    print(f"Formula set: {set_id}")
    print(f"Seed: {seed}")
    print(f"Formula search dir: {base_dir}")
    print(f"Set dir [{_status(set_dir)}]: {set_dir}")
    print(f"Seed dir [{_status(seed_dir)}]: {seed_dir}")
    print(f"Target data dir [{_status(data_dir)}]: {data_dir}")

    file_map = {
        set_dir / "audio_features_high_level.csv": data_dir
        / "audio_features_high_level.csv",
        set_dir / "labels_appears_in_region.csv": data_dir
        / "labels_appears_in_region.csv",
        set_dir / "train_table.csv": data_dir / "train_table.csv",
        seed_dir / "model_metrics.json": data_dir / "model_metrics.json",
        seed_dir / "model.joblib": data_dir / "model.joblib",
        seed_dir / "test_predictions.csv": data_dir / "test_predictions.csv",
        seed_dir / "error_rows.csv": data_dir / "error_rows.csv",
        seed_dir / "error_counts_by_region.csv": data_dir
        / "error_counts_by_region.csv",
        seed_dir / "feature_importance.csv": data_dir / "feature_importance.csv",
        seed_dir / "model_metadata.json": data_dir / "model_metadata.json",
    }

    print("\nStep 1/4: Copy canonical region metrics")
    # copy region metrics to canonical destination first
    _copy(seed_dir / "region_metrics.csv", data_dir / "region_metrics.csv")

    print(f"\nStep 2/4: Copy core artifacts ({len(file_map)} files)")
    copied = 0
    for src, dst in file_map.items():
        if dst.name in {"region_metrics.csv", "mvp_region_metrics.csv"}:
            continue
        copied += 1
        print(f"[{copied}/{len(file_map)}] Preparing copy")
        _copy(src, dst)

    if apply_country_profile:
        print("\nStep 3/4: Rebuild + retrain with country profile")
        _rebuild_with_country_profile(
            data_dir=data_dir,
            charts_csv=charts_csv,
            features_csv=features_csv,
            track_catalog_csv=track_catalog_csv,
            high_level_features_csv=data_dir / "audio_features_high_level.csv",
            genres_csv=genres_csv,
            nonviral_meta_csv=nonviral_meta_csv,
            country_profile_csv=country_profile_csv,
            chart_name=chart_name,
            seed=seed,
            cv_folds=cv_folds,
            cv_jobs=cv_jobs,
            rf_jobs=rf_jobs,
        )
    else:
        print("\nStep 3/4: Skip country profile rebuild (flag not set)")

    print("\nStep 4/4: Sync compatibility alias + baseline snapshot")
    # keep UI compatibility alias synced to canonical region metrics
    _copy(data_dir / "region_metrics.csv", data_dir / "mvp_region_metrics.csv")

    _snapshot_baseline(data_dir)

    elapsed = time.perf_counter() - started
    print("\nPromotion complete.")
    print(f"Selected formula set: {set_id}")
    print(f"Selected seed: {seed}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print("You can now run: uv run python src/make_visualizations.py")
    print("And launch UI: uv run streamlit run src/app/user_interface.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a formula-search run to canonical src/data artifacts"
    )
    parser.add_argument(
        "--set", dest="set_id", required=True, help="Formula set ID, e.g. A"
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed, e.g. 42")
    parser.add_argument(
        "--formula-search-dir",
        default="src/data/formula_search",
        help="Base formula search directory",
    )
    parser.add_argument(
        "--data-dir", default="src/data", help="Canonical data directory"
    )
    parser.add_argument("--charts-csv", default="src/data/charts.csv")
    parser.add_argument("--features-csv", default="src/data/audio_features.csv")
    parser.add_argument("--track-catalog-csv", default="src/data/track_catalog.csv")
    parser.add_argument("--genres-csv", default=None)
    parser.add_argument(
        "--nonviral-meta-csv", default="src/data/nonviral_track_ids.csv"
    )
    parser.add_argument("--chart-name", default="top200")
    parser.add_argument(
        "--apply-country-profile",
        action="store_true",
        help="After promotion, rebuild train table with country_profile_features.csv and retrain canonical model",
    )
    parser.add_argument(
        "--country-profile-csv", default="src/data/country_profile_features.csv"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="CV folds during country-profile retrain (lower uses less memory)",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Parallel jobs for cross-validation during retrain",
    )
    parser.add_argument(
        "--rf-jobs",
        type=int,
        default=1,
        help="Parallel jobs for RandomForest during retrain",
    )
    args = parser.parse_args()

    promote(
        set_id=args.set_id,
        seed=args.seed,
        base_dir=Path(args.formula_search_dir),
        data_dir=Path(args.data_dir),
        charts_csv=Path(args.charts_csv),
        features_csv=Path(args.features_csv),
        track_catalog_csv=Path(args.track_catalog_csv),
        genres_csv=args.genres_csv,
        nonviral_meta_csv=args.nonviral_meta_csv,
        chart_name=args.chart_name,
        apply_country_profile=args.apply_country_profile,
        country_profile_csv=Path(args.country_profile_csv),
        cv_folds=args.cv_folds,
        cv_jobs=args.cv_jobs,
        rf_jobs=args.rf_jobs,
    )


if __name__ == "__main__":
    main()
