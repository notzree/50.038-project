import argparse
import gc
import os
from pathlib import Path

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import pandas as pd


def _write_dataframe_csv_chunked(
    df: pd.DataFrame,
    path: str,
    *,
    chunk_rows: int = 350_000,
    label: str = "rows",
) -> None:
    """Write CSV in chunks to avoid long silent hangs and some native crashes on huge frames."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n <= chunk_rows:
        print(f"Writing {label} ({n} rows) to {path} ...", flush=True)
        df.to_csv(path, index=False)
        print(f"Finished writing {path}", flush=True)
        return

    print(
        f"Writing {label} ({n} rows) to {path} in chunks of {chunk_rows} ...",
        flush=True,
    )
    for i, start in enumerate(range(0, n, chunk_rows)):
        stop = min(start + chunk_rows, n)
        chunk = df.iloc[start:stop]
        chunk.to_csv(path, mode="w" if i == 0 else "a", header=(i == 0), index=False)
        print(f"  ... {label}: rows {start}-{stop - 1} / {n - 1}", flush=True)
    print(f"Finished writing {path}", flush=True)


def build_labels_and_train_table(
    charts_csv: str,
    features_csv: str,
    track_catalog_csv: str | None,
    high_level_features_csv: str | None,
    labels_csv: str,
    train_csv: str,
    chart_name: str,
    genres_csv: str | None = None,
    nonviral_meta_csv: str | None = None,
) -> None:
    print("Building labels and training table")
    print(f"Charts CSV: {charts_csv}")
    print(f"Features CSV: {features_csv}")
    if track_catalog_csv:
        print(f"Track catalog CSV: {track_catalog_csv}")
    if high_level_features_csv:
        print(f"High-level features CSV: {high_level_features_csv}")
    if genres_csv:
        print(f"Genres CSV: {genres_csv}")
    if nonviral_meta_csv:
        print(f"Non-viral meta CSV: {nonviral_meta_csv}")

    print(f"Loading features CSV: {features_csv} ...", flush=True)
    features = pd.read_csv(features_csv, on_bad_lines="warn", low_memory=False)

    if "status" in features.columns:
        features = features[features["status"] == "ok"].copy()

    print(f"Using {len(features)} feature rows for train table build", flush=True)

    tracks = features[["track_id"]].drop_duplicates()
    print(f"Unique tracks: {len(tracks)}", flush=True)

    print(f"Loading charts: {charts_csv} ...", flush=True)
    charts = pd.read_csv(charts_csv)
    charts = charts[charts["chart"] == chart_name].copy()
    charts["track_id"] = (
        charts["url"].str.split("?").str[0].str.rsplit("/", n=1).str[-1]
    )

    regions = charts[["region"]].drop_duplicates()
    print(f"Regions in chart filter: {len(regions)}", flush=True)

    # Cross product of all tracks x all regions
    tracks["_key"] = 1
    regions["_key"] = 1
    print("Building track x region label grid (merge)...", flush=True)
    pairs = tracks.merge(regions, on="_key").drop(columns="_key"])
    print(f"Label grid rows: {len(pairs)}", flush=True)

    # Positive labels: (track_id, region) that appear in charts
    positives = charts[["track_id", "region"]].drop_duplicates()
    positives["appears_in_region"] = 1

    print("Merging positives into label grid...", flush=True)
    labels = pairs.merge(positives, on=["track_id", "region"], how="left")
    labels["appears_in_region"] = labels["appears_in_region"].fillna(0).astype(int)
    print("Sorting labels...", flush=True)
    labels = labels.sort_values(["track_id", "region"]).reset_index(drop=True)

    labels_shape = labels.shape
    label_counts = labels["appears_in_region"].value_counts().sort_index()

    _write_dataframe_csv_chunked(labels, labels_csv, label="labels")

    del pairs, positives, tracks, regions, charts
    gc.collect()

    drop_cols = [c for c in ["status", "error", "file_path"] if c in features.columns]
    features_clean = features.drop(columns=drop_cols)
    del features
    gc.collect()
    print(
        f"Features matrix for merge: {len(features_clean)} rows x "
        f"{len(features_clean.columns)} cols",
        flush=True,
    )

    # Merge catalog-level metadata (source + global nonviral flag)
    if track_catalog_csv:
        try:
            catalog_cols = ["track_id", "source_type", "global_nonviral"]
            catalog = pd.read_csv(track_catalog_csv)
            keep_cols = [c for c in catalog_cols if c in catalog.columns]
            if "track_id" in keep_cols:
                catalog = catalog[keep_cols].drop_duplicates(
                    subset=["track_id"], keep="first"
                )
                features_clean = features_clean.merge(
                    catalog, on="track_id", how="left"
                )
                if "global_nonviral" in features_clean.columns:
                    features_clean["global_nonviral"] = (
                        features_clean["global_nonviral"].fillna(0).astype(int)
                    )
                if "source_type" in features_clean.columns:
                    features_clean["source_type"] = features_clean[
                        "source_type"
                    ].fillna("unknown")
                print("Merged track catalog metadata into features", flush=True)
                del catalog
                gc.collect()
        except Exception as e:
            print(f"Warning: could not merge track catalog metadata: {e}")

    if high_level_features_csv:
        try:
            high_level = pd.read_csv(high_level_features_csv)
            high_level = high_level.drop_duplicates(subset=["track_id"], keep="first")
            before = len(features_clean)
            features_clean = features_clean.merge(high_level, on="track_id", how="left")
            n_missing = int(
                features_clean.filter(regex="_proxy$").isna().all(axis=1).sum()
            )
            print(
                f"Merged high-level features: {before} rows, "
                f"tracks missing proxy features={n_missing}"
            )
            for version_col in [
                "high_level_features_version",
                "human_features_version",
            ]:
                if version_col in features_clean.columns:
                    features_clean = features_clean.drop(columns=[version_col])
            del high_level
            gc.collect()
        except Exception as e:
            print(f"Warning: could not merge high-level features: {e}")

    print(
        f"Merging labels ({len(labels)} rows) with features ({len(features_clean)} rows)...",
        flush=True,
    )
    train = labels.merge(features_clean, on="track_id", how="inner")
    print(f"Train matrix after merge: {len(train)} rows", flush=True)
    del labels, features_clean
    gc.collect()

    # Merge genre features if available
    if genres_csv:
        try:
            genres = pd.read_csv(genres_csv)[["track_id", "primary_genre"]]
            train = train.merge(genres, on="track_id", how="left")
            train["primary_genre"] = train["primary_genre"].fillna("unknown")
            n_with_genre = (train["primary_genre"] != "unknown").sum()
            print(f"Merged genres: {n_with_genre}/{len(train)} rows have genre data", flush=True)
            del genres
            gc.collect()
        except Exception as e:
            print(f"Warning: could not merge genres: {e}")

    # Merge non-viral metadata if available (legacy path)
    if nonviral_meta_csv:
        try:
            if "global_nonviral" in train.columns:
                print(
                    "Skipping legacy non-viral merge; global_nonviral already present",
                    flush=True,
                )
            else:
                nonviral = pd.read_csv(nonviral_meta_csv)[
                    ["track_id", "global_nonviral"]
                ]
                train = train.merge(nonviral, on="track_id", how="left")
                train["global_nonviral"] = (
                    train["global_nonviral"].fillna(0).astype(int)
                )
                n_nonviral = (train["global_nonviral"] == 1).sum()
                print(
                    f"Merged non-viral metadata: {n_nonviral}/{len(train)} rows are non-viral",
                    flush=True,
                )
                del nonviral
                gc.collect()
        except Exception as e:
            print(f"Warning: could not merge non-viral metadata: {e}")

    # Sanity check: global non-viral should not appear as positive
    if {"global_nonviral", "appears_in_region"}.issubset(train.columns):
        contradictions = train[
            (train["global_nonviral"] == 1) & (train["appears_in_region"] == 1)
        ]
        if len(contradictions) > 0:
            n_tracks = contradictions["track_id"].nunique()
            print(
                "Warning: found global non-viral tracks with positive regional labels "
                f"({len(contradictions)} rows, {n_tracks} tracks)",
                flush=True,
            )
        else:
            print(
                "Sanity check passed: no global non-viral positives found",
                flush=True,
            )

    _write_dataframe_csv_chunked(train, train_csv, label="train table")

    print(f"Wrote labels to {labels_csv} with shape={labels_shape}", flush=True)
    print(label_counts, flush=True)
    print(f"Wrote train table to {train_csv} with shape={train.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels and train table")
    parser.add_argument(
        "--charts", default="src/data/charts.csv", help="Path to charts CSV"
    )
    parser.add_argument(
        "--features",
        default="src/data/audio_features.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--high-level-features",
        default="src/data/audio_features_high_level.csv",
        help="Optional path to high-level features CSV",
    )
    parser.add_argument(
        "--human-features",
        default=None,
        help="Deprecated alias for --high-level-features",
    )
    parser.add_argument(
        "--track-catalog",
        default="src/data/track_catalog.csv",
        help="Optional path to track catalog CSV",
    )
    parser.add_argument(
        "--labels-out",
        default="src/data/labels_appears_in_region.csv",
        help="Output path for labels CSV",
    )
    parser.add_argument(
        "--train-out",
        default="src/data/train_table.csv",
        help="Output path for training table CSV",
    )
    parser.add_argument(
        "--chart-name",
        default="top200",
        help="Chart name to filter (default: top200)",
    )
    parser.add_argument(
        "--genres",
        default=None,
        help="Optional path to genre_features.csv (from fetch_genres.py)",
    )
    parser.add_argument(
        "--nonviral-meta",
        default=None,
        help="Optional path to nonviral_track_ids.csv (from download_nonviral.py)",
    )
    args = parser.parse_args()

    build_labels_and_train_table(
        charts_csv=args.charts,
        features_csv=args.features,
        track_catalog_csv=args.track_catalog,
        high_level_features_csv=args.high_level_features or args.human_features,
        labels_csv=args.labels_out,
        train_csv=args.train_out,
        chart_name=args.chart_name,
        genres_csv=args.genres,
        nonviral_meta_csv=args.nonviral_meta,
    )
    print("Dataset build complete")


if __name__ == "__main__":
    main()
