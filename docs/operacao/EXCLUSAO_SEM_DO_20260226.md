# Exclusao Sem Do (Auditavel)

Data: 2026-02-27  
Status: lote A + lote B1 executados

## Base de evidencia

- Inventario de scripts: `results/ops/scripts_inventory/20260226T224326Z/summary.json`
- Auditoria de referencias: `results/ops/repo_cleanup/20260225T140523Z/script_reference_audit.csv`
- Regra usada: candidato "sem do" = `reference_hits=0` e classe `research` ou `maintenance`.

## Lote A (remocao imediata segura)

Estes arquivos nao aparecem em chamadas do fluxo oficial nem em outros scripts.

Execucao: concluida em 2026-02-26.

- `scripts/bench/evaluate_auto_regime_model.py`
- `scripts/bench/train_auto_regime_model.py`
- `scripts/maintenance/__init__.py`
- `scripts/maintenance/archive_legacy_docs.py`
- `scripts/maintenance/clean_figs.py`
- `scripts/maintenance/clean_old_results.py`
- `scripts/maintenance/clean_tmp.py`
- `scripts/research/bot_whatsapp.py`
- `scripts/research/build_graph3d_dataset.py`
- `scripts/research/index_results.py`
- `scripts/research/robo_filtrador.py`
- `scripts/research/run_metric_health.py`
- `scripts/research/run_metric_logistics.py`
- `scripts/research/simulate_rolling_forecasts.py`
- `scripts/research/train_classifier.py`

## Lote B (forte candidato, revisar 1x antes de apagar)

Pastas de pesquisa que nao entram no fluxo diario oficial:

- `scripts/sim/`
- `scripts/engine/`
- `docs/notes/`

Execucao parcial:

- `docs/historico/` removido em 2026-02-27 (bloco B1, apenas arquivo historico).
- Pendentes: `scripts/sim/`, `scripts/engine/`, `docs/notes/`.

## Nao remover

Nucleo de producao:

- `scripts/lab/run_corr_macro_offline.py`
- `scripts/ops/run_daily_master.py`
- `scripts/ops/run_daily_validation.py`
- `scripts/ops/run_daily_sector_alerts.py`
- `scripts/ops/build_sector_structural_report.py`
- `scripts/ops/publish_latest_if_gate_ok.py`
- `engine/`
- `config/lab_corr_policy.json`
- `contracts/`
- `tests/`

## Execucao recomendada

1. Remover apenas o `Lote A`.
2. Rodar:
   - `python3 -m pytest -q`
   - `bash ./scripts/ops/run_repo_healthcheck.sh`
3. Se verde, abrir `Lote B` em blocos (B1 concluido).
