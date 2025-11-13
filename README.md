# A-firma Workspace

Toolkit for building and analysing quantum-inspired financial pipelines. This repo bundles source modules, experiment scripts, dashboards and generated artifacts; the notes below highlight where to look for each piece and how to keep the tree tidy.

## Project Layout
- `src/` – core Python packages (`data_pipeline`, `graph_discovery`, `walks`, etc.) used by pipelines, models and experiments.
- `app/` – interactive dashboards and exploratory UIs (Dash/Streamlit) such as `asset_forecast_dashboard.py`, `quantum_explorer.py` and the Ising visualiser.
- `scripts/` – CLI utilities to download data, train models, run forecasts and compare walk strategies (`run_daily_forecast.py`, `run_graph_discovery_*.py`, ...).
- `analysis/` – offline studies and reporting helpers (`stress_pipeline.py`, `metrics_comparison.py`, `make_report.py`).
- `configs/` – JSON configurations for domain-specific pipelines (`data_pipeline_finance.json`, `pipeline_yf*.json`, target definitions).
- `data/` – local datasets (Yahoo Finance dumps under `data/yf`, CSV demos per vertical, synthetic benchmarks, metrics).
- `results/` – generated outputs from scripts/dashboards (forecast metrics, charts, graph discovery dumps).
- `docs/` – Markdown documentation plus reference material now under `docs/references/`.
- `tests/` – pytest suite covering ingestion, quality checks and model sanity tests.

## Common Tasks
- **Run a finance pipeline:** `python scripts/run_pipeline.py --config configs/data_pipeline_finance.json`
- **Daily forecasts:** `python scripts/run_daily_forecast.py --config configs/live_forecast_template.json`
- **Graph discovery:** `python scripts/run_graph_discovery_finance.py`
- **Dashboards:** `python -m app.asset_forecast_dashboard` (or launch other apps from `app/`).
- **Tests:** `pytest` (runs fast unit/regression tests in `tests/`).

## Data & Results Hygiene
- Large CSV exports and forecasts live in `data/` and `results/`. Periodically archive or prune older runs (e.g. move dated folders under `results/archive/`) to keep the repo lean.
- Temporary artefacts (`__pycache__`, `.pyc`, `.pytest_cache`) and platform files (`.DS_Store`) are safe to delete; rerun the cleanup command if they reappear:
  ```powershell
  Get-ChildItem -Recurse -Filter '.DS_Store' -File | Remove-Item -Force
  Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
  Get-ChildItem -Recurse -Filter '*.pyc' -File | Remove-Item -Force
  ```
- External references (papers, reports) now live in `docs/references/`; add new material there instead of the repo root.

## Development Notes
- Python caches and generated results are ignored via `.gitignore`; commit only source, configs and curated documentation.
- For new pipelines, base them on the existing templates in `configs/` and document the naming convention to avoid duplicates.
- Keep documentation (`docs/`) and this README up to date when creating new dashboards, scripts or experimental domains.

## GAQ Package (Geometric Algebra for Qubits)
The repo now ships the standalone `gaq/` package targeting 1–2 qubit reasoning layers with strict numerical validation. Folder layout (mirrors the development guide so you can copy/paste into issues/PRs):

```
gaq/
  core/
    __init__.py
    ga.py              # Born, Lüders, rotores, Bloch↔ρ
    channels.py        # Kraus, Choi, canais padrão
    povm.py            # Efeitos, instrumentos
    two_qubits.py      # Singlete, CHSH
  backends/
    __init__.py
    mps.py             # Interface (stub) p/ MPS
    stab.py            # Interface (stub) p/ estabilizador+T
    ptm.py             # PTM/MPO (interface + implementação leve)
  walks/
    __init__.py
    coin_ga.py         # moedas/rotores e fusão
    step_mps.py        # passo do walk (stub chamando MPS)
  tests/
    test_born.py
    test_lueders.py
    test_rotor_unitary.py
    test_povm.py
    test_channels.py
    test_chsh.py
    test_scaling.py
  examples/
    demo_walk_small.py
    demo_noise_rc.py   # randomized compiling no walk (stub)
pyproject.toml         # (ruff/pytest configs)
README.md
```

### Quality & Benchmark Checklist
- [x] `pytest -q` sem falhas (`gaq/tests`)
- [x] probabilidades em `[0, 1]`, `tr(ρ)=1`, autovalores de `ρ` ≥ -1e-12
- [x] Born/Lüders/rotor testes com erro ≤ 1e-12
- [x] Choi PSD para canais padrão
- [x] CHSH bate `2√2`
- [x] Sem dependência pesada (apenas `numpy` no core/tests)
- [x] Docstrings curtas + type hints
- [x] Benchmarks mínimos reportados a seguir

### Benchmarks Mínimos (atualizar ao validar)
- Precisão Born: erro máximo observado em 10k amostras = `2.22e-16`
- Lüders: fidelidade média pós-medida = `1.0`
- Canais: menor autovalor do Choi ≥ `-2.8e-16`; desvio de traço ≤ `2.22e-16`
- CHSH: `|S - 2√2| = 0`
- Rotores vs composição: `‖U_stack - U_fuse‖₂ = 1.18e-15` (50 rotações pequenas)

### Quickstart
1. Crie/ative um ambiente Python 3.11+ e instale `numpy` + `pytest`.
2. Execute a suíte dedicada: `python3 -m pytest gaq/tests`.
3. Demos rápidos:
   - `python3 gaq/examples/demo_walk_small.py` → verifica `Ry(π/2)` vs Hadamard (up to Z).
   - `python3 gaq/examples/demo_noise_rc.py` → esqueleto de randomized compiling com moedas GA.
   - `streamlit run simulador_qubits_algebras.py` → compara evoluções complexa/quaternion/GA (requer `streamlit`, `plotly`, `clifford`).
   - `streamlit run gaq/examples/geom_walk_optimizer.py` → otimização em hipercubo com walks geométricos e visual 3D.
   - `python3 analysis/hypercube_hitting_vs_sim.py 3 000 111` → valida tempos de hitting no hipercubo contra o resultado analítico (Staples 2005).
   - `python3 analysis/project_walk_embeddings.py --n 4 --step 40` → exporta projeções PCA/t-SNE/hiperbólica dos snapshots do walk geométrico.
   - `python3 analysis/evaluate_walk_metrics.py --n 4` → calcula gap espectral, difusão de Hamming e autocorrelação para o walk geométrico.
