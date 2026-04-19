# First-Time Setup and Runbook

This guide is for someone cloning the project for the first time, including large local audio collections (for example, ~100,000 files).

## What this project does

- Builds a region-level classifier on `(track_id, region)`.
- Target label: `appears_in_region`.
  - `1` if track appears at least once in that region's `top200` chart.
  - `0` otherwise.
- Uses extracted audio features + high-level proxy features for training.

## 0) Prerequisites

- Python 3.12+
- `uv` installed
- Enough disk for audio + CSV outputs

From repo root:

```bash
uv sync
```

## 1) Put audio files in expected folder

- Folder: `src/data/songs/`
- Naming: `<track_id>.mp3`

Optional quick count:

```bash
ls "src/data/songs" | wc -l
```

## 2) Ensure charts dataset exists

If `src/data/charts.csv` is missing:

```bash
uv run python src/main.py --limit 1
```

## 3) Run full pipeline (recommended)

Using existing songs:

```bash
uv run python src/run_pipeline.py
```

With download step:

```bash
uv run python src/run_pipeline.py --download --limit 100
```

Pipeline creates:

1. `src/data/audio_manifest.csv`
2. `src/data/track_catalog.csv`
3. `src/data/audio_qc_summary.json`
4. `src/data/audio_qc_issues.csv`
5. `src/data/audio_features.csv`
6. `src/data/audio_features_high_level.csv`
7. `src/data/labels_appears_in_region.csv`
8. `src/data/train_table.csv`
9. `src/data/model_metrics.json`
10. `src/data/model.joblib`
11. `src/data/model_metadata.json`
12. `src/data/region_metrics.csv`
13. `src/data/test_predictions.csv`
14. `src/data/error_rows.csv`
15. `src/data/error_counts_by_region.csv`
16. visualizations in `src/data/plots/`

Important visual outputs include:

- `src/data/plots/01_dataset_overview.png`
- `src/data/plots/04_model_comparison.png`
- `src/data/plots/05_confusion_matrices.png`
- `src/data/plots/08_region_feature_lift_heatmap.png`
- `src/data/plots/09_region_probability_gap.png`

## 4) Large-scale run strategy (~100k audio)

Recommended:

1. Sanity run on smaller subset first.
2. Confirm QC + extraction success + training outputs.
3. Run full set overnight.

## 5) Predict a single audio file

Interactive prompt:

```bash
uv run python src/predict_single_audio.py
```

Direct path:

```bash
uv run python src/predict_single_audio.py --audio "/full/path/to/song.mp3" --region Singapore --model src/data/model.joblib
```

Finder picker (if available):

```bash
uv run python src/predict_single_audio.py --pick-file --region Singapore --model src/data/model.joblib
```

## 6) Minimum done checklist

- `uv sync` completed
- `src/data/songs/` populated
- `src/data/charts.csv` exists
- `uv run python src/run_pipeline.py` completes
- `src/data/model_metrics.json` exists
- single-song prediction runs successfully

## 7) Optional experimentation

Try alternative high-level-feature formula sets (kept separate from core runbook flow):

```bash
uv run python src/optimize_high_level_formulas.py --formula-sets A,B,C,D --seeds 42 --output-dir src/data/formula_search
```
