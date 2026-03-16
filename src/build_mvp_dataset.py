import argparse

import polars as pl


def build_labels_and_train_table(
    charts_csv: str,
    features_csv: str,
    labels_csv: str,
    train_csv: str,
    chart_name: str,
) -> None:
    print("Building MVP labels and training table")
    print(f"Charts CSV: {charts_csv}")
    print(f"Features CSV: {features_csv}")

    features = pl.read_csv(features_csv)

    if "status" in features.columns:
        features = features.filter(pl.col("status") == "ok")

    tracks = features.select("track_id").unique()

    charts_lf = (
        pl.scan_csv(charts_csv)
        .filter(pl.col("chart") == chart_name)
        .with_columns(
            pl.col("url")
            .str.split("?")
            .list.first()
            .str.split("/")
            .list.last()
            .alias("track_id")
        )
    )

    regions = charts_lf.select("region").unique().collect()

    positives = (
        charts_lf.select(["track_id", "region"])
        .unique()
        .with_columns(pl.lit(1).alias("appears_in_region"))
        .collect()
    )

    pairs = tracks.join(regions, how="cross")

    labels = (
        pairs.join(positives, on=["track_id", "region"], how="left")
        .with_columns(pl.col("appears_in_region").fill_null(0).cast(pl.Int8))
        .sort(["track_id", "region"])
    )

    labels.write_csv(labels_csv)

    drop_cols = [c for c in ["status", "error", "file_path"] if c in features.columns]
    features_clean = features.drop(drop_cols)

    train = labels.join(features_clean, on="track_id", how="inner")
    train.write_csv(train_csv)

    label_counts = labels.group_by("appears_in_region").len().sort("appears_in_region")
    print(f"Wrote labels to {labels_csv} with shape={labels.shape}")
    print(label_counts)
    print(f"Wrote train table to {train_csv} with shape={train.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVP labels and train table")
    parser.add_argument(
        "--charts", default="src/data/charts.csv", help="Path to charts CSV"
    )
    parser.add_argument(
        "--features",
        default="src/data/audio_features_basic.csv",
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
    args = parser.parse_args()

    build_labels_and_train_table(
        charts_csv=args.charts,
        features_csv=args.features,
        labels_csv=args.labels_out,
        train_csv=args.train_out,
        chart_name=args.chart_name,
    )
    print("MVP dataset build complete")


if __name__ == "__main__":
    main()
