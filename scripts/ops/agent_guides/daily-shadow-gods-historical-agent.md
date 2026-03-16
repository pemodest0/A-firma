# Daily Shadow Gods Historical Agent

Objetivo:
- recalcular diariamente o replay histórico sem futuro dos quatro deuses
- produzir o bloco `desempenho histórico simulado` para `2023`, `2024` e `2025`
- registrar ordens recomendadas, fills simulados, dias sem operação e NAV por cenário

Entradas:
- `config/shadow_gods_portfolios.json`
- `data/raw/finance/yfinance_daily`

Saídas:
- `results/ops/agents/daily_shadow_gods_historical/latest_summary.json`
- `website-ui/public/data/site/latest_shadow_gods_historical.json`
- CSVs públicos em `website-ui/public/data/site/shadow_gods_historical/...`

Regras:
- não usar o próprio dia como sinal; o driver deve entrar deslocado em um dia
- manter os quatro deuses congelados
- preservar três blocos de capital: `200`, `1000`, `10000`
- expor contagem exata de ordens, fills, trade days e no-trade days
