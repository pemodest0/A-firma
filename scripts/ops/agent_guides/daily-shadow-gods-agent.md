---
display_name: Daily Shadow Gods Agent
mission: Simular diariamente os quatro shadow gods congelados em três blocos de capital e registrar ordens, fills e estados.
run_order: 4
depends_on: [daily-operation-agent]
consumes: [results/ops/agents/daily_operation/latest_summary.json, results/ops/agents/daily_vigilance/latest_summary.json, data/raw/finance/yfinance_daily]
produces: [results/ops/agents/daily_shadow_gods/latest_summary.json]
rules: [nao_mascarar_fail, congelar_4_gods, persistir_ordens_e_fills, permitir_no_trade_quando_o_regime_pedir]
---

# Daily Shadow Gods Agent

Roda os quatro shadows principais congelados:
- Apollo
- Zeus
- Hephaestus
- Hermes

Cada deus mantém três blocos de capital:
- `R$200`
- `R$1.000`
- `R$10.000`

O agente:
- lê o `daily_operation`
- traduz o regime do dia para um estado executável
- gera ordens simuladas ou `no-trade`
- simula fills com fricção
- atualiza estado e histórico por cenário
- publica um resumo consolidado para a UI e para os outros agentes
