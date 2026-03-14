---
display_name: Agente diário de operação
mission: Recalcular os modos oficiais, a confiança e o modo recomendado com base na melhor base local disponível.
run_order: 3
depends_on: [daily-ingestion-agent, daily-backfill-agent]
consumes: [data/raw/finance/yfinance_daily, results/ops/agents/daily_ingestion/latest_summary.json]
produces: [results/ops/agents/daily_operation/latest_summary.json]
rules: [uma_unica_verdade_operacional, sem_usar_dado_velho_como_atual]
---

Constrói a leitura oficial do motor e atualiza os modos exibidos no produto.
