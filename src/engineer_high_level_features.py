import argparse
from pathlib import Path

import pandas as pd

from high_level_features import HIGH_LEVEL_FEATURE_COLUMNS, compute_high_level_features


def build_high_level_features(input_csv: str, output_csv: str) -> None:
    print("Building high-level feature proxies")
    print(f"Input features: {input_csv}")
    print(f"Output high-level features: {output_csv}")

    df = pd.read_csv(input_csv)
    if "track_id" not in df.columns:
        raise ValueError("Input features CSV must contain track_id column")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    rows = []
    for _, row in df.iterrows():
        feats = compute_high_level_features(row)
        rows.append({"track_id": row["track_id"], **feats})

    out_df = pd.DataFrame(rows, columns=["track_id", *HIGH_LEVEL_FEATURE_COLUMNS])
    out_df = out_df.drop_duplicates(subset=["track_id"], keep="first")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {output_csv} with shape={out_df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer high-level audio features")
    parser.add_argument(
        "--input",
        default="src/data/audio_features.csv",
        help="Path to raw extracted features CSV",
    )
    parser.add_argument(
        "--output",
        default="src/data/audio_features_high_level.csv",
        help="Path to output high-level features CSV",
    )
    args = parser.parse_args()

    build_high_level_features(args.input, args.output)


if __name__ == "__main__":
    main()
