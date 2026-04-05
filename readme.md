# Project Workflow

New here? Start with `FIRST_TIME_SETUP_AND_RUNBOOK.md` for first-time setup and large-audio guidance.

## Setup
```bash
uv sync
```

## One-command pipeline

Use existing songs in `src/data/songs/`:
```bash
uv run python src/run_pipeline.py
```

Download songs first, then run everything:
```bash
uv run python src/run_pipeline.py --download --limit 100
```

Pipeline outputs:
- `src/data/audio_manifest.csv`
- `src/data/audio_qc_summary.json`
- `src/data/audio_qc_issues.csv`
- `src/data/audio_features.csv`
- `src/data/audio_features_human.csv`
- `src/data/labels_appears_in_region.csv`
- `src/data/train_table.csv`
- `src/data/model_metrics.json`
- `src/data/model.joblib`
- `src/data/model_metadata.json`
- `src/data/region_metrics.csv`
- `src/data/test_predictions.csv`
- `src/data/error_rows.csv`
- `src/data/error_counts_by_region.csv`
- `src/data/plots/` visualizations

Key visualization files:
- `src/data/plots/01_dataset_overview.png`
- `src/data/plots/02_label_distribution.png`
- `src/data/plots/03_feature_distributions.png`
- `src/data/plots/04_model_comparison.png`
- `src/data/plots/05_confusion_matrices.png`
- `src/data/plots/06_region_performance.png`
- `src/data/plots/07_region_errors.png`
- `src/data/plots/08_region_feature_lift_heatmap.png`
- `src/data/plots/09_region_probability_gap.png`

Run pipeline + single-song prediction in one command:
```bash
uv run python src/run_pipeline.py --predict-audio "/full/path/to/song.mp3" --predict-region Singapore
```

## Single-song prediction

Direct path:
```bash
uv run python src/predict_single_audio.py --audio src/data/songs/<track_id>.mp3 --region Singapore --model src/data/model.joblib
```

Interactive prompt mode:
```bash
uv run python src/predict_single_audio.py
```

Finder picker mode (if available in Python build):
```bash
uv run python src/predict_single_audio.py --pick-file --region Singapore --model src/data/model.joblib
```

## Optional manual steps

Extract full features:
```bash
uv run python src/extract_features.py --manifest src/data/audio_manifest.csv --output src/data/audio_features.csv --feature-set full
```

Build human-readable proxy features:
```bash
uv run python src/engineer_human_features.py --input src/data/audio_features.csv --output src/data/audio_features_human.csv
```

Build labels + train table:
```bash
uv run python src/build_dataset.py
```

Train model:
```bash
uv run python src/train_model.py
```

Run multi-seed benchmark:
```bash
uv run python src/benchmark_models.py --seeds 42,7,123
```

Generate visualization set:
```bash
uv run python src/make_visualizations.py
```

This generates the region-aware high-level feature visuals as:
- `08_region_feature_lift_heatmap.png`
- `09_region_probability_gap.png`
