# Assyntrax Platform + Eigen Engine

Repositorio canonico da plataforma Assyntrax e do motor estrutural Eigen Engine.

## Links oficiais
- Repositorio: `https://github.com/pemodest0/Assyntrax`
- Branch oficial: `main`
- App oficial (Vercel): `https://assyntrax.vercel.app`
- Licenca: MIT (`LICENSE`)

## Nomenclatura oficial
- `Assyntrax`: empresa/plataforma/site.
- `Eigen Engine`: motor estrutural.

## Escopo atual
- Motor estrutural causal (correlacao rolling + espectro + regime + gate).
- Pipeline diario auditavel com publicacao condicionada ao gate.
- API/site operacional em Next.js para consumo dos artefatos.

## Estrutura principal
- `scripts/lab/run_corr_macro_offline.py`: nucleo do motor estrutural (gera regime, diagnosticos e gate).
- `config/lab_corr_policy.json`: politica oficial de parametros.
- `scripts/ops/run_daily_master.py`: pipeline diario principal.
- `scripts/ops/publish_latest_if_gate_ok.py`: publicacao condicionada ao gate.
- `engine/structural/`: modulos estruturais reutilizaveis (RMT, espectro, CSD, score, impacto).
- `engine/core/universe.py`: selecao deterministica de universo global/setorial.
- `engine/ops/metadata.py`: contrato de metadata de ativos.
- `website-ui/`: diretorio tecnico local do site Assyntrax (nome de pasta legado).
- `results/`: artefatos de execucao e validacao.

## Fluxo canonico por sessao
1. `git fetch origin --prune`
2. `git pull --ff-only origin main`
3. `./scripts/ops/run_repo_healthcheck.sh`
4. Implementar escopo
5. `cd website-ui && npm run build` (quando mexer no site)
6. Commit pequeno e objetivo
7. `git push origin main`

## Comandos essenciais
- Sincronizar local com remoto (remoto vence):
  - Mac/Linux: `./scripts/ops/git_sync_canonical.sh`
  - Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\git_sync_canonical.ps1`
- Rodar pipeline diario local:
  - Mac/Linux: `bash ./scripts/ops/run_daily_jobs.sh 23 80`
  - Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\run_daily_jobs.ps1 -Seed 23 -MaxAssets 80`

## Dados e artefatos
- Layout canonico de dados: `docs/operacao/DATA_LAYOUT_CANONICO.md`
- Snapshot de validacao: `results/validation/latest_validation.json`
- Snapshot de plataforma: `results/platform/latest_db_snapshot.json`

## Mapa tecnico (arquivos e funcoes)
- Mapa detalhado do motor: `docs/motor/EIGEN_ENGINE_FILE_FUNCTION_MAP.md`
- Manual mestre: `docs/motor/MANUAL_MESTRE_ASSYNTRAX.md`
- Teoria: `docs/motor/THEORY_ASSYNTRAX.md`

## Operacao e qualidade
- Healthcheck: `docs/operacao/REPO_HEALTHCHECK.md`
- Checklist diario: `docs/operacao/CHECKLIST_OPERACAO_EIGEN_ENGINE.md`
- Governanca GitHub canonico: `docs/operacao/GITHUB_CANONICO.md`

## Licenca
Este projeto esta licenciado sob MIT. Veja `LICENSE`.
