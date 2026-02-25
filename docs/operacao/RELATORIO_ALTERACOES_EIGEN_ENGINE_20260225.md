# Relatorio Consolidado de Alteracoes - Assyntrax / Eigen Engine

Data: 2026-02-25  
Base remota: `origin/main`  
HEAD de referencia (inicio da rodada): `4944a95`  
HEAD de formalizacao (fim da rodada): `88afea3`

## 1. Objetivo

Consolidar as alteracoes tecnicas recentes para preparar o lancamento oficial com nomenclatura formal:

- Empresa/plataforma: `Assyntrax`
- Motor: `Eigen Engine`

## 2. Marcos de evolucao (ultimos meses)

Principais commits consolidados (ordem cronologica recente):

- `2026-02-23` `4944a95`: suite de ground truth estrutural + runbook Mac para pytest/DNS.
- `2026-02-23` `cab2ec3`: stack estrutural v1 (RMT, spectral, CSD, Forman, score, contracts).
- `2026-02-23` `6d98129`: consolidacao teorica no `THEORY_ASSYNTRAX.md`.
- `2026-02-23` `e27834d`: ingestao canonica de energia (ONS) one-shot.
- `2026-02-22` `2b2d4f3`: relatorio de clareza setorial + anti-leakage + treino temporal model-c.
- `2026-02-22` `9cfb47f`: remocao de legado `verdict` e endurecimento de execucao de universo.
- `2026-02-22` `5545fff`: robustez do pipeline diario pesado e fluxo STATUS canonico.
- `2026-02-21` `e36b02f`: entrega da plataforma de copiloto + checkpoint model-c GNN.
- `2026-02-21` `e2b6842`: contexto operacional canonico (commit de referencia informado).
- `2026-02-21` `0ecae21`: enforce de sincronizacao canonica Git.
- `2026-02-21` `7766cea` / `6921247` / `84139d9`: endurecimento do app, rota `/app/venda` e promocao do estado Mac como canonico.

## 3. Escopo funcional consolidado

### 3.1 Motor (Eigen Engine)

- Nucleo operacional: `scripts/lab/run_corr_macro_offline.py`.
- Politica oficial: `config/lab_corr_policy.json`.
- Metricas estruturais e regime: `p1`, `deff`, overlap, bootstrap, histerese e gate de publicacao.
- Diagnostico hierarquico (global + setorial) e impacto estrutural (ativo/setor/global) em trilha atual.

### 3.2 Operacao diaria

- Orquestracao: `scripts/ops/run_daily_master.py` + `scripts/ops/run_daily_jobs.sh`.
- Publicacao condicionada: `scripts/ops/publish_latest_if_gate_ok.py`.
- Healthcheck de repositorio: `scripts/ops/run_repo_healthcheck.sh`.

### 3.3 Site/API

- Frontend operacional: `website-ui/`.
- Rotas API: `website-ui/app/api/**`.
- Paginas-chave: `/app/dashboard`, `/app/setores`, `/app/operacao`, `/app/venda`, `/app/teoria`, `/app/aplicacoes`, `/app/casos`.

## 4. Formalizacao de nomenclatura (rodada 2026-02-25)

Documentos/base ajustados para regra oficial:

- `README.md`
- `docs/ENGINE_FREEZE.md`
- `docs/INDEX.md`
- `docs/AI_SYSTEM_PROMPT.md`
- `docs/KNOWLEDGE_ASSYNTRAX.md`
- `docs/motor/MANUAL_MESTRE_ASSYNTRAX.md`
- `docs/motor/LIVRO_MOTOR_ASSYNTRAX_300P.md`
- `docs/motor/THEORY_ASSYNTRAX.md`
- `docs/motor/README.md`
- `website-ui/app/(site)/page.tsx`
- `website-ui/app/(site)/product/page.tsx`
- `website-ui/app/(site)/about/page.tsx`
- `website-ui/app/(site)/pt/about/page.tsx`
- `website-ui/app/(site)/layout.tsx`
- `website-ui/app/app/layout.tsx`
- `website-ui/app/app/dashboard/page.tsx`
- `website-ui/components/CopilotChat.tsx`
- `website-ui/components/sections/HeroSection.tsx`
- `website-ui/components/sections/UseCasesSection.tsx`
- `website-ui/public/assets/og/eigen-engine-og.svg`
- `docs/operacao/NOMENCLATURA_OFICIAL_ASSYNTRAX_EIGEN_ENGINE.md` (novo)

## 5. Estado do manual do motor

Manual oficial ativo:

- `docs/motor/MANUAL_MESTRE_ASSYNTRAX.md`

Estado atual do manual:

- titulo atualizado para `Manual Mestre do Eigen Engine (Assyntrax)`;
- secao explicita de nomenclatura oficial;
- estrutura tecnica preservada (dados -> metrica -> regime -> gate -> API -> limites);
- compatibilidade mantida com path legado para evitar quebra.

## 6. Validacoes executadas nesta rodada

- `git fetch origin --prune`
- `git pull --ff-only origin main`
- `bash ./scripts/ops/run_repo_healthcheck.sh` (resultado: `failed_checks=0`, run: `20260225T120126Z`)
- `npm run build` em `website-ui/` (build de producao OK)
- `python3 -m pytest -q` (resultado: `80 passed, 1 skipped`)
- `bash ./scripts/ops/run_repo_healthcheck.sh` novamente no estado final (resultado: `failed_checks=0`, run: `20260225T122346Z`)

## 7. Pendencias objetivas para fechamento de release

- Rodar scan periodico de termos legados por `rg` antes de cada deploy.
- Congelar copy final de marketing (home/product/about) com revisao unica.
- Validar build final do site apos qualquer ajuste de copy/metadata.
- Manter referencia temporal explicita (`data_last_date`) em toda resposta da IA.
