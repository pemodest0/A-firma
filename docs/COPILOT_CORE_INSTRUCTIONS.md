# Eigen Engine Assistant Core Instructions (B + C + Gate)

## Objetivo
Padronizar o comportamento do Eigen Engine Assistant (copiloto do Assyntrax) para operar com governanca, causalidade e rastreabilidade.

## Arquivo canonico de instrucoes
- `config/copilot_instruction_core.v1.json`

Este arquivo define:
- Identidade permanente do produto.
- Diferenciais tecnicos obrigatorios na comunicacao.
- Constituicao (regras que nunca podem ser violadas).
- Contrato de resposta (secoes e limites).

## Fluxo operacional
1. Pipeline diario roda `scripts/ops/run_daily_jobs.ps1`.
2. No final, roda `scripts/ops/build_copilot_shadow.py --run-id <run_id>`.
3. O script gera:
- `results/ops/copilot/<run_id>/shadow_summary.json`
- `results/ops/copilot/<run_id>/executive_summary.txt`
- `results/ops/copilot/<run_id>/technical_summary.md`
- `results/ops/copilot/latest_release.json`
4. `website-ui` le o shadow e responde no endpoint `/api/copilot`.
5. UI conversa com o copiloto em `/app/copiloto`.
6. `build_platform_db.py` indexa run + copiloto em SQLite para historico operacional.

## Modelo B (shadow)
- Usa `models/auto_regime_model.joblib` quando disponivel.
- Predicao e risco por run sao gravados no shadow.
- Se modelo nao carregar, aplica fallback heuristico documentado no proprio artefato.

## Modelo C (shadow)
- Usa proxy estrutural temporal sobre `risk_truth_panel.json`.
- Gera `risk_score`, `confidence`, `regime` e `publish_ready`.
- Fica no mesmo fluxo de gate e integridade da publicacao.

## Gate de publicacao
`publishable=true` somente se:
- Integridade do run e artefatos estiver OK.
- Gate operacional do run estiver liberado.
- Modelo C estiver `publish_ready=true`.

Se qualquer condicao falhar:
- status do copiloto: `NAO PUBLICAVEL`
- resposta fica em modo diagnostico

## Banco operacional
- Arquivo: `results/platform/assyntrax_platform.db`
- Tabelas: `runs`, `asset_signals`, `copilot_runs`
- Snapshot para site/API: `results/platform/latest_db_snapshot.json`

## Contrato de resposta do chat
1. O que mudou
2. Mecanismo provavel
3. Evidencias
4. Limites e incerteza
5. O que monitorar amanha

## Regras constitucionais
- Nunca prometer retorno.
- Nunca recomendar compra/venda.
- Sempre separar risco estrutural de confianca.
- Sempre declarar politica ativa e causalidade.
- Se integridade falhar, nao publicar.
