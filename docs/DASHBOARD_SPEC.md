# Dashboard Spec (Atual)

## Objetivo
Mostrar estado operacional real, nao previsao promocional.

## Blocos por ativo
1. Estado atual (`regime`).
2. Confiabilidade (`confidence`, `quality`, `data_adequacy`).
3. Motivo do gate (`reason`, `status`).

## Regras de UI
- `validated`: sinal exibido como operacional.
- `watch`: alerta exibido com cautela.
- `inconclusive`: mostrar apenas diagnostico; esconder acao.

## Fontes de dados
- `results/ops/snapshots/<run_id>/api_snapshot.jsonl`
- `results/ops/snapshots/<run_id>/summary.json`
- `results/validation/VERDICT.json`
- `results/validation/risk_truth_panel.json`

## Rotas de API esperadas
- `/api/run/latest`
- `/api/assets`
- `/api/assets/[asset]`
- `/api/risk-truth`

## Rotas de app publicadas (2026-03-01)
- `/app/dashboard`: visão operacional principal do universo financeiro.
- `/app/financas` e `/app/finance`: leitura detalhada de finanças (mesmo painel).
- `/app/energia`: leitura setorial de energia.
- `/app/agro`: dashboard mensal Agro BR (estado, evidência, impacto e acoplamento).
- `/app/evidencias`: contexto histórico e simulações de uso do motor.
- `/app/copiloto`: chat do Eigen Engine Assistant com contexto dos artefatos.

## Segurança web aplicada
- Content Security Policy, X-Frame-Options, X-Content-Type-Options.
- Referrer-Policy e Permissions-Policy restritivas.
- Endpoints operacionais de chat e plataforma com `Cache-Control: no-store`.
