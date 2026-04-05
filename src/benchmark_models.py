import argparse
import json
from pathlib import Path

import pandas as pd

from train_model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed model benchmark")
    parser.add_argument(
        "--input",
        default="src/data/train_table.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--seeds",
        default="42,7,123",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--output-dir",
        default="src/data/benchmarks",
        help="Directory for per-seed outputs and summary",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test split",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    print(f"Running benchmark for seeds: {seeds}")
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Seed {seed} ---")
        metrics = train_and_evaluate(
            input_csv=args.input,
            metrics_out=str(seed_dir / "model_metrics.json"),
            model_out=str(seed_dir / "model.joblib"),
            region_metrics_out=str(seed_dir / "region_metrics.csv"),
            test_preds_out=str(seed_dir / "test_predictions.csv"),
            errors_out=str(seed_dir / "error_rows.csv"),
            test_size=args.test_size,
            seed=seed,
        )

        selected = metrics["selected_model"]
        selected_scores = metrics["models"][selected]
        tuned = metrics["thresholding"]["val_best_f1_threshold"]

        # Use test metrics for reporting
        test_scores = selected_scores.get("test", selected_scores)

        summary_rows.append(
            {
                "seed": seed,
                "selected_model": selected,
                "selected_f1": test_scores["f1"],
                "selected_roc_auc": test_scores["roc_auc"],
                "selected_precision": test_scores["precision"],
                "selected_recall": test_scores["recall"],
                "best_threshold": tuned["threshold"],
                "best_threshold_f1": tuned["f1"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("seed")
    summary_csv = out_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg = {
        "num_runs": int(len(summary_rows)),
        "mean_selected_f1": float(summary_df["selected_f1"].mean()),
        "std_selected_f1": float(summary_df["selected_f1"].std(ddof=0)),
        "mean_selected_roc_auc": float(summary_df["selected_roc_auc"].mean()),
        "mean_best_threshold": float(summary_df["best_threshold"].mean()),
    }
    agg_json = out_dir / "benchmark_aggregate.json"
    agg_json.write_text(json.dumps(agg, indent=2))

    print("\nBenchmark complete")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote aggregate: {agg_json}")
    print(summary_df)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
