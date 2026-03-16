# First-Time Setup and Runbook

This guide is for someone cloning the project for the first time, including the case where you already have a **very large local audio collection** (for example, ~100,000 files from charting songs).

## What this project does (MVP)

- Builds an MVP classifier for `(track_id, region)`.
- Target label: `appears_in_region`
  - `1` if a track appears at least once in that region's `top200` chart.
  - `0` otherwise.
- Uses basic extracted audio features + region to train baseline models.

---

## 0) Prerequisites

- Python 3.12+
- `uv` installed
- Sufficient disk space (100k audio files + generated CSVs/model artifacts)
- If using built-in downloader (`src/main.py`): network access + ffmpeg support for `yt-dlp`

From repository root:

```bash
uv sync
```

---

## 1) Put audio files in the expected folder

Expected folder:

- `src/data/songs/`

Expected filename convention:

- `<track_id>.mp3`

Where `track_id` matches Spotify URL tail used in charts data, e.g.:

- `https://open.spotify.com/track/6mICuAdrwEjh6Y6lroV2Kg` -> `6mICuAdrwEjh6Y6lroV2Kg.mp3`

If you already have many files (e.g. 100k), place them in `src/data/songs/` before running pipeline.

Optional quick count:

```bash
ls "src/data/songs" | wc -l
```

---

## 2) Ensure charts dataset exists

If `src/data/charts.csv` is not present, run:

```bash
uv run python src/main.py --limit 1
```

This will initialize charts download and fetch one clip. If you already have `charts.csv`, skip.

---

## 3) Run full MVP pipeline (recommended)

Using existing audio files in `src/data/songs/`:

```bash
uv run python src/run_mvp_pipeline.py
```

This runs:

1. Build manifest (`src/data/audio_manifest.csv`)
2. Extract audio features (`src/data/audio_features_basic.csv`)
3. Build labels (`src/data/labels_appears_in_region.csv`)
4. Build training table (`src/data/train_table_mvp.csv`)
5. Train/evaluate models (`src/data/mvp_metrics.json`)
6. Save selected model (`src/data/mvp_model.joblib`)
7. Save analysis artifacts:
   - `src/data/mvp_region_metrics.csv`
   - `src/data/mvp_test_predictions.csv`
   - `src/data/mvp_error_rows.csv`
   - `src/data/mvp_error_counts_by_region.csv`
8. Generate initial label distribution visualization:
   - `src/data/plots/mvp_label_distribution.png`

If you want pipeline to download songs too:

```bash
uv run python src/run_mvp_pipeline.py --download --limit 100
```

If you also want a single-song prediction at the end of the pipeline:

```bash
uv run python src/run_mvp_pipeline.py --predict-audio "/full/path/to/song.mp3" --predict-region Singapore
```

---

## 4) For very large audio sets (~100k): practical run strategy

Recommended approach:

1. Run once on a subset first (sanity):
   - Move/copy a small subset into `src/data/songs/`
   - Run pipeline and confirm artifacts look correct
2. Then run full set overnight/long session.

Notes for scale:

- Feature extraction is the longest step.
- CSV outputs can become large; keep enough free disk.
- If process is interrupted, rerun pipeline; it will rebuild manifest and regenerate downstream outputs.

---

## 5) Benchmark stability (recommended before reporting)

Run multiple seeds:

```bash
uv run python src/benchmark_mvp.py --seeds 42,7,123
```

Outputs:

- `src/data/benchmarks/benchmark_summary.csv`
- `src/data/benchmarks/benchmark_aggregate.json`

---

## 6) Predict a single audio file

Interactive prompt mode (asks for path and region):

```bash
uv run python src/predict_single_audio.py
```

Direct path mode:

```bash
uv run python src/predict_single_audio.py --audio "/full/path/to/song.mp3" --region Singapore --model src/data/mvp_model.joblib
```

Popup picker mode (if available in your Python build):

```bash
uv run python src/predict_single_audio.py --pick-file --region Singapore --model src/data/mvp_model.joblib
```

If popup fails with Tkinter error, use interactive/direct path mode.

---

## 7) Useful references in this repo

- Workflow commands: `commands_runbook.txt`
- MVP target definition: `mvp_target_notes.txt`
- Limitations/roadmap: `mvp_limitations_and_roadmap.txt`
- Week 8 speaking guide: `week8_talk_track.txt`

---

## 8) Minimum "done" checklist for first-time runner

- `uv sync` completed
- `src/data/songs/` populated
- `src/data/charts.csv` exists
- `uv run python src/run_mvp_pipeline.py` completes
- `src/data/mvp_metrics.json` exists and is readable
- `uv run python src/predict_single_audio.py` runs successfully
