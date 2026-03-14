---
display_name: Agente diário de qualidade
mission: Medir cobertura, frescor e poda potencial do universo observado.
run_order: 5
depends_on: [daily-ingestion-agent, daily-backfill-agent, daily-operation-agent]
consumes: [data/raw/finance/yfinance_daily, results/ops/agents/daily_ingestion/latest_summary.json]
produces: [results/ops/agents/daily_data_quality/latest_summary.json]
rules: [separar_critico_nucleo_periferia, qualidade_publica_e_rastreavel]
---

Classifica atrasos e prioriza o que precisa de cobertura melhor ou revisão de universo.
