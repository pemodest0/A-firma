# Finance Product Ready Workflow (Assyntrax / Eigen Engine)

Objetivo: fechar o pacote operacional de Financas para publicacao interna/externa sem improviso.

## Script oficial

```bash
python3 scripts/ops/build_finance_product_ready_pack.py \
  --run-dir results/lab_corr_macro/20260224T022339Z \
  --impact-dir results/lab_corr_macro/20260224T022339Z/hierarchical/impact_learning_2015_2026_compare \
  --alert-budget 0.15 \
  --alert-budget-sweep 0.10,0.15,0.20 \
  --alert-dedupe-days 20 \
  --lead-window-days 30 \
  --min-event-gap-days 20 \
  --ai-outdir results/ops/ai_knowledge \
  --outdir results/ops/finance_product_ready
```

## O que o script gera

- Avaliacao historica estrutural:
  - `historical_structure_summary.json`
  - `historical_structure_next_month_indication.json`
  - `historical_structure_stress_prealert_budget_sweep.csv`
- Brief operacional para IA:
  - `results/ops/ai_knowledge/latest_operational_brief.json`
  - `results/ops/ai_knowledge/operational_brief_<timestamp>.json`
- Relatorio de prontidao:
  - `results/ops/finance_product_ready/finance_product_ready_<timestamp>.json`
  - `results/ops/finance_product_ready/finance_product_ready_<timestamp>.md`
  - `results/ops/finance_product_ready/latest_finance_product_ready.json`

## Regras de leitura do veredito

- `overall_readiness=pass`: pronto para operacao normal.
- `overall_readiness=warn`: pronto com ressalvas (ex.: base desatualizada).
- `overall_readiness=fail`: nao publicar, corrigir checks obrigatorios.

## Check minimo antes de publicar

1. `data_last_date` deve estar dentro de SLA operacional.
2. `next_month_status_ok=true`.
3. `stress_prealert_summary_non_empty=true`.
4. `regime_horizon_f1_available=true`.
5. `latest_operational_brief.json` sem `NaN` literal.

