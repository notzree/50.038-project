import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def make_label_distribution_plot(labels_csv: str, out_png: str) -> None:
    df = pl.read_csv(labels_csv)
    if "appears_in_region" not in df.columns:
        raise ValueError("labels CSV must contain appears_in_region column")

    counts = (
        df.group_by("appears_in_region")
        .len()
        .rename({"len": "count"})
        .sort("appears_in_region")
    )

    total = int(counts["count"].sum())
    labels = ["Not Appears (0)", "Appears (1)"]
    values = [0, 0]
    for row in counts.iter_rows(named=True):
        idx = int(row["appears_in_region"])
        if idx in (0, 1):
            values[idx] = int(row["count"])

    percentages = [(v / total * 100.0) if total > 0 else 0.0 for v in values]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#6c8ebf", "#e07a5f"])
    plt.title("MVP Label Distribution: appears_in_region")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)

    for bar, count, pct in zip(bars, values, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
        )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"Wrote initial visualization to {out_path}")
    print(f"Class 0 count: {values[0]:,} ({percentages[0]:.2f}%)")
    print(f"Class 1 count: {values[1]:,} ({percentages[1]:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make initial MVP visualization")
    parser.add_argument(
        "--labels",
        default="src/data/labels_appears_in_region.csv",
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--out",
        default="src/data/plots/mvp_label_distribution.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    make_label_distribution_plot(args.labels, args.out)


if __name__ == "__main__":
    main()
