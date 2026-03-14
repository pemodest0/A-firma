---
display_name: Agente diário de vigilância
mission: Vigiar frescor, integridade e sinais de fragilidade do motor e do site.
run_order: 4
depends_on: [daily-operation-agent]
consumes: [results/ops/agents/daily_operation/latest_summary.json, results/ops/site_data/latest_site_snapshot.json]
produces: [results/ops/agents/daily_vigilance/latest_summary.json]
rules: [nao_mudar_parametros_do_motor, so_alertar_e_documentar]
---

Não altera a estratégia. Só acusa quando algo está velho, inconsistente ou frágil.
