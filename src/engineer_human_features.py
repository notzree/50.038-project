import argparse
from pathlib import Path

import pandas as pd

from human_features import HUMAN_FEATURE_COLUMNS, compute_human_features


def build_human_features(input_csv: str, output_csv: str) -> None:
    print("Building human-readable feature proxies")
    print(f"Input features: {input_csv}")
    print(f"Output human features: {output_csv}")

    df = pd.read_csv(input_csv)
    if "track_id" not in df.columns:
        raise ValueError("Input features CSV must contain track_id column")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    rows = []
    for _, row in df.iterrows():
        feats = compute_human_features(row)
        rows.append({"track_id": row["track_id"], **feats})

    out_df = pd.DataFrame(rows, columns=["track_id", *HUMAN_FEATURE_COLUMNS])
    out_df = out_df.drop_duplicates(subset=["track_id"], keep="first")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {output_csv} with shape={out_df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engineer human-readable audio features"
    )
    parser.add_argument(
        "--input",
        default="src/data/audio_features.csv",
        help="Path to raw extracted features CSV",
    )
    parser.add_argument(
        "--output",
        default="src/data/audio_features_human.csv",
        help="Path to output human features CSV",
    )
    args = parser.parse_args()

    build_human_features(args.input, args.output)


if __name__ == "__main__":
    main()
