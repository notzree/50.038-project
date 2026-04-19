# Setup Notes

Use `readme.md` as the source of truth.

Minimal checks:

```bash
uv sync
uv run python src/run_pipeline.py --download --limit 10 --download-nonviral --nonviral-limit 5 --skip-visualizations
uv run python src/optimize_high_level_formulas.py --formula-sets A,B,C,D --seeds 42 --output-dir src/data/formula_search
uv run python src/make_visualizations.py
```
