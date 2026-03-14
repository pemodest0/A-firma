---
display_name: Agente diário de publicação
mission: Regerar snapshot, sincronizar artefatos públicos e publicar o site com a leitura mais recente.
run_order: 6
depends_on: [daily-operation-agent, daily-vigilance-agent, daily-data-quality-agent]
consumes: [results/ops/site_data/latest_site_snapshot.json, website-ui]
produces: [results/ops/agents/daily_publish/latest_summary.json]
rules: [publicacao_sem_estado_falso, deploy_so_com_artefato_coerente]
---

Executa a trilha de publicação e deixa rastro do deploy diário.
