import argparse

import pandas as pd


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
    country_profile_csv: str | None = None,
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
    if country_profile_csv:
        print(f"Country profile CSV: {country_profile_csv}")

    features = pd.read_csv(features_csv)

    if "status" in features.columns:
        features = features[features["status"] == "ok"].copy()

    tracks = features[["track_id"]].drop_duplicates()

    charts = pd.read_csv(charts_csv)
    charts = charts[charts["chart"] == chart_name].copy()
    charts["track_id"] = (
        charts["url"].str.split("?").str[0].str.rsplit("/", n=1).str[-1]
    )

    regions = charts[["region"]].drop_duplicates()

    # Cross product of all tracks x all regions
    tracks["_key"] = 1
    regions["_key"] = 1
    pairs = tracks.merge(regions, on="_key").drop(columns="_key")

    # Positive labels: (track_id, region) that appear in charts
    positives = charts[["track_id", "region"]].drop_duplicates()
    positives["appears_in_region"] = 1

    labels = pairs.merge(positives, on=["track_id", "region"], how="left")
    labels["appears_in_region"] = labels["appears_in_region"].fillna(0).astype(int)
    labels = labels.sort_values(["track_id", "region"]).reset_index(drop=True)

    labels.to_csv(labels_csv, index=False)

    drop_cols = [c for c in ["status", "error", "file_path"] if c in features.columns]
    features_clean = features.drop(columns=drop_cols)

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
                print("Merged track catalog metadata into features")
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
        except Exception as e:
            print(f"Warning: could not merge high-level features: {e}")

    train = labels.merge(features_clean, on="track_id", how="inner")

    # Merge country profile features if available
    if country_profile_csv:
        try:
            country = pd.read_csv(country_profile_csv)
            if "region" not in country.columns:
                raise ValueError("country profile CSV must contain 'region' column")

            exclude_cols = {"region", "iso3", "data_year", "source_notes"}
            numeric_cols = [
                c
                for c in country.columns
                if c not in exclude_cols
                and pd.api.types.is_numeric_dtype(country[c])
                and not country[c].dropna().empty
            ]

            keep_cols = ["region", *numeric_cols]
            country = country[keep_cols].drop_duplicates(
                subset=["region"], keep="first"
            )

            train = train.merge(country, on="region", how="left")

            if numeric_cols:
                missing_any = train[numeric_cols].isna().any(axis=1)
                train["country_profile_missing"] = missing_any.astype(int)

                for col in numeric_cols:
                    median = country[col].dropna().median()
                    fill_value = float(median) if pd.notna(median) else 0.0
                    train[col] = train[col].fillna(fill_value)

                print(
                    "Merged country profile features: "
                    f"cols={numeric_cols}, rows with missing before fill={int(missing_any.sum())}"
                )
            else:
                print(
                    "Country profile CSV found but no numeric feature columns detected"
                )
        except Exception as e:
            print(f"Warning: could not merge country profile features: {e}")

    # Merge genre features if available
    if genres_csv:
        try:
            genres = pd.read_csv(genres_csv)[["track_id", "primary_genre"]]
            train = train.merge(genres, on="track_id", how="left")
            train["primary_genre"] = train["primary_genre"].fillna("unknown")
            n_with_genre = (train["primary_genre"] != "unknown").sum()
            print(f"Merged genres: {n_with_genre}/{len(train)} rows have genre data")
        except Exception as e:
            print(f"Warning: could not merge genres: {e}")

    # Merge non-viral metadata if available (legacy path)
    if nonviral_meta_csv:
        try:
            if "global_nonviral" in train.columns:
                print(
                    "Skipping legacy non-viral merge; global_nonviral already present"
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
                    f"Merged non-viral metadata: {n_nonviral}/{len(train)} rows are non-viral"
                )
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
                f"({len(contradictions)} rows, {n_tracks} tracks)"
            )
        else:
            print("Sanity check passed: no global non-viral positives found")

    train.to_csv(train_csv, index=False)

    label_counts = labels["appears_in_region"].value_counts().sort_index()
    print(f"Wrote labels to {labels_csv} with shape={labels.shape}")
    print(label_counts)
    print(f"Wrote train table to {train_csv} with shape={train.shape}")


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
        help="Optional path to genre_features.csv",
    )
    parser.add_argument(
        "--nonviral-meta",
        default=None,
        help="Optional path to nonviral_track_ids.csv (from download_nonviral.py)",
    )
    parser.add_argument(
        "--country-profile",
        default=None,
        help="Optional path to country_profile_features.csv (joined by region)",
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
        country_profile_csv=args.country_profile,
    )
    print("Dataset build complete")


if __name__ == "__main__":
    main()
