# Graph Engine Plan (Prompt 1)

## Reuse map (what already exists)

### Takens embedding + phase utilities
- `spa/engine/diagnostics/regime_labels.py`: `RegimeClassifier.embed`, `select_embedding`, recurrence + local features.
- `spa/models/takens_knn.py`: `TakensKNN` + `embed` helper.
- `spa/diagnostics_phase.py`: embedding segmentation + plotting patterns.
- `scripts/legacy/engine/run_global_multivariate.py`: `takens_multivariate` helper legado.

### Data loading (finance + generic)
- `scripts/finance/run_daily_forecast.py`: `_load_series` (CSV or yfinance) and standardized pricing series.
- `scripts/finance/run_financial_analysis.py`: `load_price_csv`, `load_value_csv`, `download_price_series`.
- `scripts/finance/yf_fetch_or_load.py`: local cache loader + yfinance download.
- `spa/io.py`: `load_dataset` for generic CSV (non-finance).

### Sanity / anti-leakage
- `scripts/bench/run_phase_benchmark.py`: flags for walk-forward + anti-leakage checks.
- `scripts/bench/run_finance_walkforward.py`: run_id, walk-forward splitting, baseline comparisons.
- `scripts/finance/run_daily_forecast.py`: baseline comparisons and output conventions.

### Reporting & plots
- `spa/run.py`: `generate_report` for PDF.
- `scripts/legacy/sim/run_lorenz_analysis.py` / `run_vanderpol_analysis.py` / `run_pendulo_duplo_analysis.py`: report.md/pdf patterns legados.
- `scripts/report/generate_figures.py`: plot utilities and embedding plots.

### Existing output schemas
- `results/*/overview.json`, `report.md`, `report.pdf`
- `results/*/api_records.jsonl` + `api_records.csv` via `spa/api_records.py`
- `results/dashboard/overview.json` (built by `scripts/report/build_dashboard_overview.py`)

## Proposed parallel path (no changes to existing motor)

### New folder (parallel engine)
```
graph_engine/
  __init__.py
  embedding.py          # wrappers around Takens (reuse)
  microstates.py        # KMeans microstates
  graph_builder.py      # transition counts + P matrix
  metastable.py         # spectral clustering for regimes
  labels.py             # STABLE/TRANSITION/UNSTABLE/NOISY rules
  export.py             # write results/latest_graph/*
  merge_existing.py     # optional merge from existing outputs
```

### New results output
```
results/latest_graph/
  universe_weekly.json
  universe_daily.json
  assets/
    {ASSET}_{TF}.json
    {ASSET}_{TF}_embedding.csv
    {ASSET}_{TF}_micrograph.json
    {ASSET}_{TF}_regimes.csv
    {ASSET}_{TF}_transitions.json
    {ASSET}_{TF}_plots/
```

### New CLI scripts (parallel run_*)
```
scripts/bench/run_graph_regime_universe.py
```

## Reuse strategy (imports only)
- Embedding: use `spa.models.takens_knn.embed` or `RegimeClassifier.embed`.
- Walk-forward/sanity: copy patterns from `scripts/bench/run_finance_walkforward.py`.
- CSV/plot exporting: mirror `scripts/legacy/sim/*analysis.py` structure.

## Non-conflict rule
- No changes to `spa/` or existing `scripts/`.
- No touching `results/latest/`.
- All new outputs go to `results/latest_graph/`.
