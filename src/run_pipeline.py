import argparse
from pathlib import Path

import pandas as pd


def _safe_read_csv(csv_path: str | None) -> pd.DataFrame | None:
    if not csv_path:
        return None
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def build_catalog_and_manifest(
    songs_dir: Path,
    manifest_path: Path,
    catalog_path: Path,
    charts_csv: str,
    nonviral_meta_csv: str | None,
) -> int:
    print("\n=== Build track catalog + manifest ===")
    print(f"Songs dir: {songs_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Catalog: {catalog_path}")

    songs = sorted(songs_dir.glob("*.mp3"))
    base = pd.DataFrame(
        [{"track_id": p.stem, "file_path": str(p)} for p in songs],
        columns=["track_id", "file_path"],
    )
    if base.empty:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(manifest_path, index=False)
        base.to_csv(catalog_path, index=False)
        print("No songs found while building catalog/manifest")
        return 0

    # Viral/chart metadata from charts dataset
    charts = _safe_read_csv(charts_csv)
    if charts is not None and {"url", "title", "artist"}.issubset(charts.columns):
        charts = charts.copy()
        charts["track_id"] = (
            charts["url"].str.split("?").str[0].str.rsplit("/", n=1).str[-1]
        )
        chart_meta = (
            charts[["track_id", "title", "artist"]]
            .dropna(subset=["track_id"])
            .drop_duplicates(subset=["track_id"], keep="first")
            .rename(columns={"title": "track_title", "artist": "track_artist"})
        )
        chart_meta["in_charts"] = 1
    else:
        chart_meta = pd.DataFrame(
            columns=["track_id", "track_title", "track_artist", "in_charts"]
        )

    # Non-viral metadata
    nonviral = _safe_read_csv(nonviral_meta_csv)
    if nonviral is not None and "track_id" in nonviral.columns:
        nonviral = nonviral.copy()
        if "global_nonviral" not in nonviral.columns:
            nonviral["global_nonviral"] = 1
        nonviral = (
            nonviral[
                [
                    c
                    for c in [
                        "track_id",
                        "global_nonviral",
                        "track_name",
                        "artists",
                        "popularity",
                    ]
                    if c in nonviral.columns
                ]
            ]
            .drop_duplicates(subset=["track_id"], keep="first")
            .rename(
                columns={
                    "track_name": "nonviral_track_name",
                    "artists": "nonviral_artists",
                }
            )
        )
    else:
        nonviral = pd.DataFrame(columns=["track_id", "global_nonviral"])

    catalog = base.merge(chart_meta, on="track_id", how="left")
    catalog = catalog.merge(nonviral, on="track_id", how="left")

    catalog["in_charts"] = catalog["in_charts"].fillna(0).astype(int)
    catalog["global_nonviral"] = catalog["global_nonviral"].fillna(0).astype(int)

    catalog["source_type"] = "unknown"
    catalog.loc[
        (catalog["in_charts"] == 1) & (catalog["global_nonviral"] == 0),
        "source_type",
    ] = "viral_charts"
    catalog.loc[
        (catalog["in_charts"] == 0) & (catalog["global_nonviral"] == 1),
        "source_type",
    ] = "global_nonviral"
    catalog.loc[
        (catalog["in_charts"] == 1) & (catalog["global_nonviral"] == 1),
        "source_type",
    ] = "mixed"

    catalog = catalog.sort_values("track_id").reset_index(drop=True)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)

    manifest = catalog[
        ["track_id", "file_path", "source_type", "global_nonviral"]
    ].rename(columns={"global_nonviral": "is_nonviral_global"})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    src_counts = catalog["source_type"].value_counts().to_dict()
    print(f"Catalog rows: {len(catalog)} | source distribution: {src_counts}")
    print(f"Manifest rows: {len(manifest)}")
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
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
        "--catalog-out",
        default="src/data/track_catalog.csv",
        help="Track catalog CSV output path",
    )
    parser.add_argument(
        "--qc-summary-out",
        default="src/data/audio_qc_summary.json",
        help="QC summary JSON output path",
    )
    parser.add_argument(
        "--qc-issues-out",
        default="src/data/audio_qc_issues.csv",
        help="QC issues CSV output path",
    )
    parser.add_argument(
        "--features-out",
        default="src/data/audio_features.csv",
        help="Audio features CSV output path",
    )
    parser.add_argument(
        "--high-level-features-out",
        "--human-features-out",
        dest="high_level_features_out",
        default="src/data/audio_features_high_level.csv",
        help="High-level feature proxy CSV output path",
    )
    parser.add_argument(
        "--feature-set",
        choices=["basic", "full"],
        default="full",
        help="Feature set to extract (default: full)",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=None,
        help="Number of parallel workers for feature extraction",
    )
    parser.add_argument(
        "--feature-max-tasks-per-child",
        type=int,
        default=100,
        help="Restart each extraction worker after N tracks to reduce memory pressure",
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
        default="src/data/train_table.csv",
        help="Training table CSV output path",
    )
    parser.add_argument(
        "--metrics-out",
        default="src/data/model_metrics.json",
        help="Model metrics JSON output path",
    )
    parser.add_argument(
        "--model-out",
        default="src/data/model.joblib",
        help="Serialized selected model output path",
    )
    parser.add_argument(
        "--region-metrics-out",
        default="src/data/region_metrics.csv",
        help="Per-region metrics CSV output path",
    )
    parser.add_argument(
        "--test-preds-out",
        default="src/data/test_predictions.csv",
        help="Test predictions CSV output path",
    )
    parser.add_argument(
        "--errors-out",
        default="src/data/error_rows.csv",
        help="Error rows CSV output path",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip all visualization generation",
    )
    parser.add_argument(
        "--viz-dir",
        default="src/data/plots",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--genres",
        default=None,
        help="Path to genre_features.csv. Auto-detected if src/data/genre_features.csv exists.",
    )
    parser.add_argument(
        "--download-nonviral",
        action="store_true",
        help="Download non-viral songs before building the manifest",
    )
    parser.add_argument(
        "--nonviral-limit",
        type=int,
        default=None,
        help="Max non-viral songs to download (requires --download-nonviral)",
    )
    parser.add_argument(
        "--nonviral-popularity-threshold",
        type=int,
        default=25,
        help="Max popularity score to qualify as non-viral (default: 25)",
    )
    parser.add_argument(
        "--nonviral-meta",
        default=None,
        help="Path to nonviral_track_ids.csv. Auto-detected if src/data/nonviral_track_ids.csv exists.",
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
    parser.add_argument(
        "--with-trends",
        action="store_true",
        help="Collect and merge Google Trends compact features into training table",
    )
    parser.add_argument(
        "--trends-output",
        default="src/data/trends_features.csv",
        help="Output path for trends feature CSV",
    )
    parser.add_argument(
        "--trends-weeks",
        type=int,
        default=12,
        help="Trailing weeks per trends query",
    )
    parser.add_argument(
        "--trends-delay",
        type=float,
        default=10.0,
        help="Delay between trends requests in seconds",
    )
    parser.add_argument(
        "--trends-max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of artist-level trends queries",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Resolve all paths relative to repo root so it works from any cwd
    def _resolve(p: str) -> str:
        return str((repo_root / p).resolve())

    songs_dir = Path(_resolve(args.songs_dir))
    manifest_path = Path(_resolve(args.manifest))
    catalog_path = Path(_resolve(args.catalog_out))
    features_out = _resolve(args.features_out)
    charts = _resolve(args.charts)
    labels_out = _resolve(args.labels_out)
    train_out = _resolve(args.train_out)
    trends_out = _resolve(args.trends_output)
    qc_summary_out = _resolve(args.qc_summary_out)
    qc_issues_out = _resolve(args.qc_issues_out)
    high_level_features_out = _resolve(args.high_level_features_out)
    metrics_out = _resolve(args.metrics_out)
    model_out = _resolve(args.model_out)
    region_metrics_out = _resolve(args.region_metrics_out)
    test_preds_out = _resolve(args.test_preds_out)
    errors_out = _resolve(args.errors_out)
    viz_dir = Path(_resolve(args.viz_dir))

    print("Starting pipeline")
    print(f"Repo root: {repo_root}")

    # --- Step 1: Download ---
    if args.download:
        print("\n=== Download songs ===")
        from download import (
            download_dataset,
            get_mp3s_for_dataset,
            unify_title_url_mappings,
        )

        csv_path = download_dataset()
        unified_df = unify_title_url_mappings(csv_path)
        get_mp3s_for_dataset(unified_df, max_workers=args.max_workers, limit=args.limit)
    else:
        print("\n=== Download songs ===")
        print("Skipping download step (use --download to enable)")

    # --- Step 1b: Download non-viral songs ---
    if args.download_nonviral:
        print("\n=== Download non-viral songs ===")
        from download_nonviral import get_nonviral_mp3s

        get_nonviral_mp3s(
            limit=args.nonviral_limit,
            max_workers=args.max_workers,
            popularity_threshold=args.nonviral_popularity_threshold,
        )
    else:
        print("\n=== Download non-viral songs ===")
        print("Skipping non-viral download step (use --download-nonviral to enable)")

    # Auto-detect nonviral metadata CSV early (for catalog build)
    nonviral_meta_path = args.nonviral_meta
    if nonviral_meta_path is None:
        auto_nonviral = repo_root / "src" / "data" / "nonviral_track_ids.csv"
        if auto_nonviral.exists():
            nonviral_meta_path = str(auto_nonviral)
            print(f"Auto-detected non-viral metadata: {nonviral_meta_path}")

    # --- Step 2: Build catalog + manifest ---
    songs_dir.mkdir(parents=True, exist_ok=True)
    song_count = build_catalog_and_manifest(
        songs_dir=songs_dir,
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        charts_csv=charts,
        nonviral_meta_csv=nonviral_meta_path,
    )
    if song_count == 0:
        print("No songs found. Stopping pipeline.")
        print("Add songs to src/data/songs or rerun with --download")
        return

    # --- Step 2b: QC manifest ---
    print("\n=== Run dataset QC (manifest) ===")
    from qc_audio_dataset import run_qc

    run_qc(
        manifest_csv=str(manifest_path),
        features_csv=None,
        summary_out=qc_summary_out,
        issues_out=qc_issues_out,
    )

    # --- Step 3: Extract features ---
    print("\n=== Extract features ===")
    from extract_features import run_extraction

    run_extraction(
        manifest_csv=str(manifest_path),
        output_csv=features_out,
        feature_set=args.feature_set,
        workers=args.feature_workers,
        max_tasks_per_child=args.feature_max_tasks_per_child,
    )

    # --- Step 3b: QC with extraction status ---
    print("\n=== Run dataset QC (post-feature extraction) ===")
    run_qc(
        manifest_csv=str(manifest_path),
        features_csv=features_out,
        summary_out=qc_summary_out,
        issues_out=qc_issues_out,
    )

    # --- Step 3c: Build high-level proxy features ---
    print("\n=== Build high-level feature proxies ===")
    from engineer_high_level_features import build_high_level_features

    build_high_level_features(
        input_csv=features_out,
        output_csv=high_level_features_out,
    )

    # --- Step 4: Build labels and train table ---
    print("\n=== Build labels and train table ===")
    from build_dataset import build_labels_and_train_table

    # Auto-detect genre CSV
    genres_path = args.genres
    if genres_path is None:
        auto_genres = repo_root / "src" / "data" / "genre_features.csv"
        if auto_genres.exists():
            genres_path = str(auto_genres)
            print(f"Auto-detected genre features: {genres_path}")

    build_labels_and_train_table(
        charts_csv=charts,
        features_csv=features_out,
        track_catalog_csv=str(catalog_path),
        high_level_features_csv=high_level_features_out,
        labels_csv=labels_out,
        train_csv=train_out,
        chart_name=args.chart_name,
        genres_csv=genres_path,
        nonviral_meta_csv=nonviral_meta_path,
    )

    # --- Step 4b: Optional trends feature collection + merge ---
    if args.with_trends:
        print("\n=== Collect and merge Google Trends features ===")
        from google_trends_collector import (
            build_trends_features,
            merge_trends_into_train_table,
        )

        build_trends_features(
            charts_path=Path(charts),
            manifest_path=Path(manifest_path),
            output_path=Path(trends_out),
            trailing_weeks=args.trends_weeks,
            delay=args.trends_delay,
            max_pairs=args.trends_max_pairs,
        )

        merged_train_out = str(Path(train_out).with_name("train_table_with_trends.csv"))
        merge_trends_into_train_table(
            train_table_path=Path(train_out),
            trends_path=Path(trends_out),
            output_path=Path(merged_train_out),
        )
        train_out = merged_train_out
        print(f"Using trends-merged train table: {train_out}")
    else:
        print("\n=== Google Trends features ===")
        print("Skipping trends stage (use --with-trends to enable)")

    # --- Step 5: Train and evaluate ---
    print("\n=== Train and evaluate models ===")
    from train_model import train_and_evaluate

    train_and_evaluate(
        input_csv=train_out,
        metrics_out=metrics_out,
        model_out=model_out,
        region_metrics_out=region_metrics_out,
        test_preds_out=test_preds_out,
        errors_out=errors_out,
        test_size=0.2,
        seed=42,
    )

    # --- Step 6: Visualizations ---
    if not args.skip_visualizations:
        print("\n=== Generate visualizations ===")
        from make_visualizations import (
            plot_confusion_matrices,
            plot_dataset_overview,
            plot_feature_distributions,
            plot_label_distribution,
            plot_model_comparison,
            plot_region_errors,
            plot_region_performance,
        )

        viz_dir.mkdir(parents=True, exist_ok=True)

        error_counts_path = Path(errors_out).with_name("error_counts_by_region.csv")

        plot_dataset_overview(
            Path(train_out), Path(labels_out), viz_dir / "01_dataset_overview.png"
        )
        plot_label_distribution(Path(labels_out), viz_dir / "02_label_distribution.png")
        plot_feature_distributions(
            Path(train_out), viz_dir / "03_feature_distributions.png"
        )
        plot_model_comparison(Path(metrics_out), viz_dir / "04_model_comparison.png")
        plot_confusion_matrices(
            Path(metrics_out), viz_dir / "05_confusion_matrices.png"
        )
        plot_region_performance(
            Path(region_metrics_out), viz_dir / "06_region_performance.png"
        )
        if error_counts_path.exists():
            plot_region_errors(error_counts_path, viz_dir / "07_region_errors.png")
    else:
        print("\n=== Visualizations ===")
        print("Skipping (--skip-visualizations enabled)")

    # --- Step 7: Optional single prediction ---
    if args.predict_audio and args.predict_region:
        print("\n=== Predict single song ===")
        from predict_single_audio import predict_single
        import json

        result = predict_single(args.predict_audio, args.predict_region, model_out)
        print(json.dumps(result, indent=2))

        if args.predict_output_json:
            out = Path(args.predict_output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
    elif args.predict_audio or args.predict_region:
        print(
            "\nSkipping prediction: provide both --predict-audio and --predict-region"
        )

    print("\nPipeline complete")
    print(f"Features: {features_out}")
    print(f"High-level features: {high_level_features_out}")
    print(f"QC summary: {qc_summary_out}")
    print(f"Labels: {labels_out}")
    print(f"Train table: {train_out}")
    print(f"Track catalog: {catalog_path}")
    print(f"Metrics: {metrics_out}")
    print(f"Model: {model_out}")


if __name__ == "__main__":
    main()
