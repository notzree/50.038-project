import argparse

import pandas as pd


def build_labels_and_train_table(
    charts_csv: str,
    features_csv: str,
    labels_csv: str,
    train_csv: str,
    chart_name: str,
    genres_csv: str | None = None,
) -> None:
    print("Building labels and training table")
    print(f"Charts CSV: {charts_csv}")
    print(f"Features CSV: {features_csv}")
    if genres_csv:
        print(f"Genres CSV: {genres_csv}")

    features = pd.read_csv(features_csv)

    if "status" in features.columns:
        features = features[features["status"] == "ok"].copy()

    tracks = features[["track_id"]].drop_duplicates()

    charts = pd.read_csv(charts_csv)
    charts = charts[charts["chart"] == chart_name].copy()
    charts["track_id"] = (
        charts["url"]
        .str.split("?").str[0]
        .str.rsplit("/", n=1).str[-1]
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

    train = labels.merge(features_clean, on="track_id", how="inner")

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
        "--labels-out",
        default="src/data/labels_appears_in_region.csv",
        help="Output path for labels CSV",
    )
    parser.add_argument(
        "--train-out",
        default="src/data/train_table_mvp.csv",
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
    args = parser.parse_args()

    build_labels_and_train_table(
        charts_csv=args.charts,
        features_csv=args.features,
        labels_csv=args.labels_out,
        train_csv=args.train_out,
        chart_name=args.chart_name,
        genres_csv=args.genres,
    )
    print("Dataset build complete")


if __name__ == "__main__":
    main()
