---
display_name: Agente diário de ingestão
mission: Atualizar o histórico local de preços com coleta remota e fallback controlado.
run_order: 1
depends_on: []
consumes: [data/raw/finance/yfinance_daily, scripts/finance/yf_fetch_or_load.py]
produces: [results/ops/agents/daily_ingestion/latest_summary.json]
rules: [sem_fallback_silencioso, historico_local_e_a_verdade]
---

Atualiza a base local diária. Não publica site, não recalibra o motor e não toma decisão de risco.
