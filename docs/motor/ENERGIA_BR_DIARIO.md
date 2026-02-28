# Energia Brasil Diario (Eigen Engine)

Fluxo espelhado do Agro, usando dados de energia do Brasil em frequencia diaria.

## 1) Ingestao oficial (ONS) com cache local

Script:

```bash
python3 scripts/energy/fetch_energy_brasil_sources.py --download-once 1
```

Datasets usados por padrao:

- `ons_carga_diaria` (carga por subsistema)
- `ons_ear_subsistema_di` (EAR diario por subsistema)
- `ons_cmo_semanal` (CMO semanal por subsistema)

Saidas:

- `results/energy_download/energy_sync_<run_id>/sync_manifest.json`
- `results/energy_download/local_pack_<run_id>/panel_long_sector.csv`
- `data/raw/ONS/ons_carga_diaria/*.csv`
- `data/raw/ONS/ons_ear_subsistema_di/*.csv`
- `data/raw/ONS/ons_cmo_semanal/*.csv`
- `results/energy_br/fetch_energy_br_<run_id>/fetch_summary.json`

## 2) Pack canonico diario

Script:

```bash
python3 scripts/energy/build_local_energy_br_pack.py
```

Saidas por run:

- `results/energy_br/local_pack_<run_id>/panel_long_sector.csv`
- `results/energy_br/local_pack_<run_id>/panel_long_energy_br.csv`
- `results/energy_br/local_pack_<run_id>/asset_metadata_energy_br.csv`
- `results/energy_br/local_pack_<run_id>/universe_fixed.csv`

Saidas canonicas:

- `data/processed/energy/br_daily/panel_long_energy_br.csv`
- `data/processed/energy/br_daily/asset_metadata_energy_br.csv`
- `data/processed/energy/br_daily/universe_fixed.csv`

## 3) Pipeline diario + latest

Script:

```bash
python3 scripts/energy/run_energy_br_daily_pipeline.py
```

Saidas latest:

- `results/energy_br/latest/hierarchical_state_latest_energy_br.json`
- `results/energy_br/latest/rankings_latest_energy_br.json`
- `results/energy_br/latest/historical_structure_summary_energy_br.json`
- `results/energy_br/latest/latest_schema_checks_energy_br.json`

Release pointer:

- `results/energy_br/latest_release_energy_br.json`

## 4) Catalogo de eventos

Arquivo:

- `config/event_catalog_energy_br.json`

Scripts de analise:

- `scripts/energy/build_energy_event_evidence.py`
- `scripts/energy/tune_energy_event_thresholds.py`
- `scripts/energy/validate_energy_temporal_blocks.py`

Regra:

- Sem artefato latest valido, API/UI deve responder indisponivel (`503`) e nao usar fallback silencioso.
