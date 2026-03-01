# Assyntrax | Eigen Engine

<p align="center">
  <img src="website-ui/public/assets/brand/assyntrax-mark.svg" alt="Logo Assyntrax" width="120" />
</p>

Repositório canônico da **Assyntrax** (empresa/plataforma) e do **Eigen Engine** (motor de diagnóstico estrutural).

## Marca e produto
- **Assyntrax**: empresa, site e plataforma operacional.
- **Eigen Engine**: motor quantitativo de regimes e transições estruturais.
- **Eigen Engine Assistant**: copiloto técnico do projeto (contextualizado por domínio).

## Setores de atuação
- **Finanças**: estrutura de risco, concentração e transição de regime.
- **Energia**: mudanças estruturais em séries de carga/custo/coupling.
- **Agro**: dinâmica macro-setorial e transições em séries mensais.

## Links oficiais
- Repositório: `https://github.com/pemodest0/Assyntrax`
- Branch oficial: `main`
- App oficial (Vercel): `https://assyntrax.vercel.app`
- Licença: MIT (`LICENSE`)

## O que o Eigen Engine faz
- Diagnóstico causal de estrutura com matriz de correlação dinâmica.
- Leitura de regime com governança de publicação por gate.
- Ranking de impacto ativo→setor e setor→global.
- Validação temporal com split treino/teste e comparação com baseline aleatório.

## Simulações e validações ativas
- Walk-forward temporal por blocos.
- Treino até data fixa e teste somente no futuro.
- Comparação de alerta estrutural vs alerta aleatório na mesma taxa.
- Controle de estabilidade entre blocos antes de promover regra/modelo.

## Estrutura principal
- `scripts/lab/run_corr_macro_offline.py`: núcleo do Eigen Engine.
- `config/lab_corr_policy.json`: política oficial de parâmetros.
- `scripts/ops/run_daily_master.py`: pipeline diário auditável.
- `scripts/ops/publish_latest_if_gate_ok.py`: publicação condicionada ao gate.
- `engine/structural/`: RMT, espectro, CSD, score e impacto.
- `engine/core/universe.py`: seleção determinística de universo global/setorial.
- `engine/ops/metadata.py`: contrato de metadados de ativos.
- `website-ui/`: diretório técnico local do site Assyntrax.
- `results/`: artefatos de execução, validação e publicação.

## Artefatos centrais
- `results/ops/finance_product_ready/latest_finance_product_ready.json`
- `results/ops/ai_knowledge/latest_operational_brief.json`
- `results/platform/latest_db_snapshot.json`
- `results/validation/latest_validation.json`

## Fluxo canônico por sessão
1. `git fetch origin --prune`
2. `git pull --ff-only origin main`
3. `./scripts/ops/run_repo_healthcheck.sh`
4. Implementar escopo
5. `cd website-ui && npm run build` (quando houver mudança de frontend)
6. Commit pequeno e objetivo
7. `git push origin main`

## Comandos essenciais
- Sincronizar local com remoto (remoto vence):
  - Mac/Linux: `./scripts/ops/git_sync_canonical.sh`
  - Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\git_sync_canonical.ps1`
- Rodar pipeline diário local:
  - Mac/Linux: `bash ./scripts/ops/run_daily_jobs.sh 23 80`
  - Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\ops\\run_daily_jobs.ps1 -Seed 23 -MaxAssets 80`

## Documentação técnica
- Manual mestre: `docs/motor/MANUAL_MESTRE_ASSYNTRAX.md`
- Teoria do motor: `docs/motor/THEORY_ASSYNTRAX.md`
- Mapa de arquivos e funções: `docs/motor/EIGEN_ENGINE_FILE_FUNCTION_MAP.md`
- Índice geral: `docs/INDEX.md`

## Licença
Este projeto está licenciado sob MIT. Consulte `LICENSE`.
