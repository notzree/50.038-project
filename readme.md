# MVP Workflow

New here? Start with `FIRST_TIME_SETUP_AND_RUNBOOK.md` for the full first-time setup and large-audio workflow.

## 1) Setup
```bash
uv sync
```

## 2) One-command full pipeline (recommended)

Use existing songs in `src/data/songs/`:
```bash
uv run python src/run_mvp_pipeline.py
```

Download songs first, then run everything:
```bash
uv run python src/run_mvp_pipeline.py --download --limit 100
```

This pipeline runs:
- manifest build (`src/data/audio_manifest.csv`)
- feature extraction (`src/data/audio_features_basic.csv`)
- label build (`src/data/labels_appears_in_region.csv`)
- train table build (`src/data/train_table_mvp.csv`)
- model training + eval (`src/data/mvp_metrics.json`)
- model artifact save (`src/data/mvp_model.joblib`)
- per-region metrics (`src/data/mvp_region_metrics.csv`)
- test predictions with probabilities (`src/data/mvp_test_predictions.csv`)
- error rows for analysis (`src/data/mvp_error_rows.csv`)
- error counts by region (`src/data/mvp_error_counts_by_region.csv`)
- initial visualization (`src/data/plots/mvp_label_distribution.png`)
- full Week 8 visualization set (`src/data/plots/week8/`)

Run pipeline + single-song prediction in one command:
```bash
uv run python src/run_mvp_pipeline.py --predict-audio "/full/path/to/song.mp3" --predict-region Singapore
```

## 3) Quick predict for one audio file (high priority)

Section 2 already creates `src/data/mvp_model.joblib`.
You can use a song from `src/data/songs/` or a new/random audio file anywhere on your device.

Use a direct file path:
```bash
uv run python src/predict_single_audio.py --audio src/data/songs/<track_id>.mp3 --region Singapore --model src/data/mvp_model.joblib
```

Or run in interactive prompt mode (it will ask for path and region):
```bash
uv run python src/predict_single_audio.py
```

Or open Finder and pick the file with a popup:
```bash
uv run python src/predict_single_audio.py --pick-file --region Singapore --model src/data/mvp_model.joblib
```
If popup mode is unavailable in your Python build, use `--audio <path>`.

Tip: predictions are most reliable when the input is similar to training clips (about 30s music segment).

Use a region name that exists in `src/data/charts.csv` (same spelling/case as training data).
To list available regions:
```bash
uv run python -c "import polars as pl; print(pl.read_csv('src/data/charts.csv').select('region').unique().sort('region'))"
```

Example output:
```json
{
  "audio_path": "src/data/songs/2oqnLprSSUkekvK1Fklj58.mp3",
  "region": "Singapore",
  "prediction": 0,
  "probability_appears_in_region": 0.0
}
```

## 4) Run steps manually (optional)

Download songs:
```bash
uv run python src/main.py
```

Download only 100 songs:
```bash
uv run python src/main.py --limit 100
```

Extract features:
```bash
uv run python src/extract_features.py --manifest src/data/audio_manifest.csv --output src/data/audio_features_basic.csv
```

Build MVP labels + train table:
```bash
uv run python src/build_mvp_dataset.py
```

Train MVP model:
```bash
uv run python src/train_mvp_model.py
```

Run 3-seed benchmark for stability:
```bash
uv run python src/benchmark_mvp.py --seeds 42,7,123
```

Benchmark outputs:
- `src/data/benchmarks/benchmark_summary.csv`
- `src/data/benchmarks/benchmark_aggregate.json`

Generate initial visualization (label distribution):
```bash
uv run python src/make_initial_visualization.py
```

Output:
- `src/data/plots/mvp_label_distribution.png`

Generate the full Week 8 visualization set:
```bash
uv run python src/make_week8_visualizations.py
```

Outputs in:
- `src/data/plots/week8/01_dataset_overview.png`
- `src/data/plots/week8/02_label_distribution.png`
- `src/data/plots/week8/03_feature_distributions.png`
- `src/data/plots/week8/04_model_comparison.png`
- `src/data/plots/week8/05_confusion_matrices.png`
- `src/data/plots/week8/06_region_performance.png`
- `src/data/plots/week8/07_region_errors.png`
