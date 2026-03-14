---
display_name: Agente diário de reconciliação
mission: Reprocessar ativos críticos e do núcleo que ficaram atrasados após a ingestão principal.
run_order: 2
depends_on: [daily-ingestion-agent]
consumes: [results/ops/agents/daily_ingestion/latest_summary.json, data/raw/finance/yfinance_daily]
produces: [results/ops/agents/daily_backfill/latest_summary.json]
rules: [foco_em_criticos_e_nucleo, sem_apagar_ativos_as_cego]
---

Tenta reduzir buracos de cobertura antes da operação oficial rodar.
