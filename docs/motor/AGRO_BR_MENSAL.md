# Agro Brasil Mensal (Eigen Engine)

Este fluxo cria o vertical Agro BR mensal sem alterar o núcleo do motor.

## 1) Ingestão

Script:

```bash
python3 scripts/agro/fetch_agro_brasil_sources.py
```

Saídas:

- `data/download/agro/**` (downloads brutos)
- `data/raw/agro/bcb/*.csv` (séries BCB mensalizadas)
- `results/agro_br/fetch_agro_br_<run_id>/fetch_summary.json`

## 2) Pack canônico mensal

Script:

```bash
python3 scripts/agro/build_local_agro_br_pack.py
```

Saídas por run:

- `results/agro_br/local_pack_<run_id>/panel_long_sector.csv`
- `results/agro_br/local_pack_<run_id>/panel_long_agro_br.csv`
- `results/agro_br/local_pack_<run_id>/asset_metadata_agro_br.csv`
- `results/agro_br/local_pack_<run_id>/universe_fixed.csv`

Saídas canônicas:

- `data/processed/agro/br_monthly/panel_long_agro_br.csv`
- `data/processed/agro/br_monthly/asset_metadata_agro_br.csv`
- `data/processed/agro/br_monthly/universe_fixed.csv`

## 3) Pipeline mensal + artefatos latest

Script:

```bash
python3 scripts/agro/run_agro_br_monthly_pipeline.py
```

Saídas latest:

- `results/agro_br/latest/hierarchical_state_latest_agro_br.json`
- `results/agro_br/latest/rankings_latest_agro_br.json`
- `results/agro_br/latest/historical_structure_summary_agro_br.json`
- `results/agro_br/latest/latest_schema_checks_agro_br.json`

Release pointer:

- `results/agro_br/latest_release_agro_br.json`

## 4) Catálogo de eventos

Arquivo:

- `config/event_catalog_agro_br.json`

Usado pelo script:

- `scripts/agro/build_agro_event_evidence.py`

## 5) API e páginas

Endpoints:

- `/api/agro/state`
- `/api/agro/rankings`
- `/api/agro/evidence`

Páginas:

- `/agro`
- `/app/agro`

Regra de gate:

- Se faltar artefato latest, API retorna `503` (sem fallback silencioso).
