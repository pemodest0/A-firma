# Assyntrax Platform + Eigen Engine

Este repositorio contem a plataforma Assyntrax, o Eigen Engine (motor de deteccao de regimes e risco), os pipelines de validacao/auditoria e o frontend operacional.

## Estado atual
- Motor unificado em `engine/` com wrappers de compatibilidade em `spa/` e `graph_engine/`.
- Pipelines operacionais em `scripts/ops/`.
- Validacoes cientificas e operacionais em `scripts/bench/validation/`.
- Site/API em `website-ui/` consumindo snapshots validados.
- Banco operacional SQLite em `results/platform/assyntrax_platform.db`.
- Artefatos canonicos em `results/ops/snapshots/<run_id>/` e `results/validation/`.

## Estrutura principal
- `engine/`: API estavel do motor (camada oficial para novos imports).
- `scripts/ops/`: rotina diaria, snapshot, contrato, drift e auditoria.
- `scripts/bench/validation/`: testes de robustez, placebo, adequacao e utilidade.
- `config/`: contrato de saida e gates versionados.
- `website-ui/`: frontend e rotas API para consumo dos artefatos.
- `results/platform/`: banco SQLite + snapshot consolidado da plataforma.
- `legacy/`: arquivos antigos ou fora do fluxo atual.

## Fluxo oficial (alto nivel)
1. Rodar jobs diarios (`scripts/ops/run_daily_jobs.ps1` no Windows, `scripts/ops/run_daily_jobs.sh` no Linux/Mac).
2. Validar contrato e gates.
3. Gerar shadow do copiloto B+C com gate (`scripts/ops/build_copilot_shadow.py`).
4. Indexar run no banco SQLite (`scripts/ops/build_platform_db.py`).
5. Publicar snapshot com `api_snapshot.jsonl` + `summary.json` + `audit_pack.json`.
6. Frontend consome ultimo run valido + status do banco.

## Criterio de pronto para producao
- Contrato de saida valido.
- Data adequacy gate aprovado.
- Drift diario sem bloqueio de deployment.
- Global status em estado aceitavel para publicacao.

## Comandos uteis
- Sincronizacao canonica (forca remoto sobre local):
  - `./scripts/ops/git_sync_canonical.sh`
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\git_sync_canonical.ps1`
- Pipeline diario:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\run_daily_jobs.ps1 -Seed 17 -MaxAssets 80`
  - `bash ./scripts/ops/run_daily_jobs.sh 23 80 "" 1`
- Copiloto shadow manual (B+C):
  - `python scripts/ops/build_copilot_shadow.py --run-id 20260210_contractfix`
- Banco SQLite manual:
  - `python scripts/ops/build_platform_db.py --run-id 20260210_contractfix`
- Launcher unico (executavel local):
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\start_platform_local.ps1 -RunPipeline`
  - `bash ./scripts/ops/start_platform_local.sh --run-pipeline`
  - `bash ./scripts/ops/start_platform_local.sh --run-pipeline --threads 1`
- Frontend local:
  - `cd website-ui`
  - `npm run dev`
  - abrir `http://localhost:3000/app/copiloto`
  - abrir `http://localhost:3000/app/plataforma`

## Nucleo de instrucoes (motor + copiloto)
- Arquivo canonico: `config/copilot_instruction_core.v1.json`.
- Regra fixa: sem promessa de retorno, sem recomendacao de compra/venda, risco separado de confianca.
- Publicacao do copiloto depende de gate + integridade (`publishable=true` no shadow).
- Artefato de shadow por run: `results/ops/copilot/<run_id>/shadow_summary.json`.
- Snapshot do banco para o site: `results/platform/latest_db_snapshot.json`.

## Telas operacionais principais
- `website-ui/app/app/dashboard/page.tsx`: painel central do motor com filtros de ativo/setor/janela/periodo.
- `website-ui/app/app/setores/page.tsx`: leitura setorial, ranking e historico de niveis.
- `website-ui/app/app/operacao/page.tsx`: rotina diaria, gate de publicacao e relatorio.
- `website-ui/app/app/venda/page.tsx`: proposta comercial, pacotes e material de reuniao.

## Documentacao recomendada
- `docs/OPS_EXECUTION_FLOW.md`
- `docs/ENGINE_GUIDE.md`
- `docs/DAILY_PIPELINE.md`
- `docs/COPILOT_CORE_INSTRUCTIONS.md`
- `docs/REPO_REFACTOR_PLAN.md`
- `docs/notes/MAC_HANDOFF.md`
- `docs/operacao/REPO_HEALTHCHECK.md`
- `docs/operacao/GITHUB_CANONICO.md`
