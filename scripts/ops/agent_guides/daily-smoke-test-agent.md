---
display_name: Agente diário de smoke test
mission: Validar se o snapshot publicado e o modo operacional continuam coerentes após a publicação.
run_order: 7
depends_on: [daily-publish]
consumes: [results/ops/site_data/latest_site_snapshot.json, results/ops/agents/daily_operation/latest_summary.json]
produces: [results/ops/agents/daily_smoke_test/latest_summary.json]
rules: [falha_visivel, sem_passar_site_incoerente]
---

Checa consistência de artefatos, datas e estado operacional. Pode usar HTTP se houver base URL configurada.
