# Project Quickstart

Main workflow (70/30 viral vs non-viral, formula search, promotion, plots, Streamlit):

```bash
uv sync

TOTAL=1000
VIRAL=$(( TOTAL * 70 / 100 ))
NONVIRAL=$(( TOTAL - VIRAL ))

uv run python src/run_pipeline.py \
  --download \
  --limit "$VIRAL" \
  --download-nonviral \
  --nonviral-limit "$NONVIRAL" \
  --skip-visualizations

uv run python src/optimize_high_level_formulas.py \
  --formula-sets A,B,C,D \
  --seeds 42 \
  --output-dir src/data/formula_search

BEST_SET=$(uv run python -c "import json;print(json.load(open('src/data/formula_search/best_formula_result.json'))['formula_set'])")
BEST_SEED=$(uv run python -c "import json;print(int(json.load(open('src/data/formula_search/best_formula_result.json'))['seed']))")

uv run python src/promote_formula_run.py \
  --set "$BEST_SET" \
  --seed "$BEST_SEED" \
  --formula-search-dir src/data/formula_search \
  --apply-country-profile \
  --country-profile-csv src/data/country_profile_features.csv

uv run python src/make_visualizations.py
uv run streamlit run src/app/user_interface.py
```
