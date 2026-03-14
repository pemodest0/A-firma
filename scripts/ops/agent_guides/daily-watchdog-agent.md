---
display_name: Agente diário de watchdog
mission: Auditar os outros agentes e tentar retries controlados quando houver falha ou estado crítico.
run_order: 8
depends_on: [daily-smoke-test-agent]
consumes: [results/ops/agents]
produces: [results/ops/agents/daily_watchdog/latest_summary.json]
rules: [retry_limitado, sem_loop_infinito, falha_irresolvida_vira_alerta]
---

É o agente de último nível. Ele não inventa novas leituras; só garante que a automação não morra silenciosamente.
