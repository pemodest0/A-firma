# Handoff Completo Assyntrax (Motor + Site + Ops)

Gerado em: 2026-02-21T22:34:29Z

## 1) Snapshot Git Atual

- Repositório: `/Users/PedroHenrique/Desktop/A-firma`
- Branch atual: `main`
- HEAD local: `fed0d6b`
- origin/main (último fetch local): `aced14b`
- Remoto canônico: `https://github.com/pemodest0/Assyntrax.git`
- Arquivos com mudança local (tracked+untracked): `116`
- Tracked modificados/deletados: `88`
- Untracked: `63`

## 2) O que já foi construído (resumo executivo)

### Motor (núcleo quantitativo)
- Núcleo operacional em `scripts/lab/run_corr_macro_offline.py` com métricas espectrais (p1, deff, overlap, bootstrap), classificação de regimes e histerese.
- Política oficial em `config/lab_corr_policy.json` (limiares, pesos, gates e parâmetros de operação).
- Artefatos operacionais e de validação gerados em `results/ops/`, `results/validation/` e sincronizados para `website-ui/public/data/...`.

### Ops (produção, governança e qualidade)
- Pipeline diário consolidado em `scripts/ops/run_daily_master.py` e wrappers de execução diária (Windows/Mac/Linux).
- Gate de publicação com bloqueio explícito (`publish_gate.json` + marcador `PUBLISH_BLOCKED`).
- Healthcheck de repositório/site em `scripts/ops/run_repo_healthcheck.sh`.
- Atualização de métricas de truth em `scripts/ops/update_prediction_truth_daily.py`.

### Site (Next.js + APIs + painéis)
- Frontend em `website-ui/` com páginas institucionais e app operacional.
- APIs internas para consumo de snapshot/regimes/setores/ativos em `website-ui/app/api/**`.
- Página teórica com fórmulas e referências em `website-ui/app/app/teoria/page.tsx`.
- Páginas de aplicações e casos reais com fontes em `website-ui/app/app/aplicacoes/page.tsx` e `website-ui/app/app/casos/page.tsx`.

### Correções mais recentes já aplicadas
- Endurecimento de estados sem dados e falhas de carregamento (setores/motor).
- Inclusão de rota `/app/venda` para remover dead-end.
- Correção de português (acentuação/cópia) e blindagem de links vazios em cards/fontes.

## 3) Linha do tempo de mudanças (todas as mudanças em Git por commit)

Formato: `hash | data | mensagem`

```text
256e4c2 | 2025-11-02 | Primeiro commit
2fc27f7 | 2025-11-13 | comitando do mac
4183500 | 2025-11-13 | Implementação das pastas e arquivos
52f589f | 2026-01-13 | feat: enhance website with SEO, accessibility, and chatbot improvements
f6657b8 | 2026-01-22 | chore: clean repo and keep spa core
a671456 | 2026-01-26 | Rename verdict engine to temporal engine and update references
f4b7c0c | 2026-01-27 | Organiza motor e pipeline de regimes + benchmarks
cc3b299 | 2026-01-28 | @codex review automatico
cc5efee | 2026-01-28 | Add website UI, routes, and sync scripts
408a90c | 2026-01-28 | Add finance pipelines, API records, and asset grouping
e100ac6 | 2026-02-03 | unify graph engine, benchmarks, and dashboard
7de30ad | 2026-02-07 | Revamp landing visuals and engine pipeline updates
2ceb8b4 | 2026-02-10 | Assyntrax: engine/site/docs/ops update
0b839d1 | 2026-02-10 | feat: update assyntrax engine+site, docs and deploy config
a2d67bf | 2026-02-10 | fix(deploy): vercel install with legacy peer deps for react19 + react-simple-maps
9d5b96f | 2026-02-10 | fix: add home route
a90073e | 2026-02-10 | fix: use site landing as root
86e1113 | 2026-02-13 | chore: snapshot completo para continuidade no Mac
5bd8bd1 | 2026-02-20 | feat(site): add pilot and sales package sections
ce2018e | 2026-02-20 | Revert "feat(site): add pilot and sales package sections"
32d3cfe | 2026-02-20 | chore(assyntrax): sync motor, setor alerts, ops and docs
257e8df | 2026-02-20 | fix(site): restore sector alerts dashboard component
d668018 | 2026-02-20 | Harden alerts auth and fix event study evaluation logic
7cc1703 | 2026-02-20 | Tighten maintenance risks in event validation and ops scripts
c34f20b | 2026-02-20 | Implement real-estate discount fallback and tighten critical exception handling
55c4cfb | 2026-02-20 | Tighten exception handling across ops maintenance scripts
7766cea | 2026-02-21 | app: harden empty-data states and add applications event study UX
6921247 | 2026-02-21 | app: add missing /app/venda route to remove dead-end
84139d9 | 2026-02-21 | sync: promote current mac website-ui state as canonical
aced14b | 2026-02-21 | docs/ui: add theory page references in app and methods site
fed0d6b | 2026-02-21 | ui: fix pt-BR copy and harden empty-link rendering
```

## 4) Mudanças locais pendentes nesta máquina (ainda não commitadas)

### 4.1 Diffstat atual

```text
 AGENT_MEMORY.md                                    |  33 -
 BUSINESS_OUTCOME.md                                |  21 -
 DEV_LOG.md                                         |  29 -
 JOURNAL.md                                         |  12 -
 MAC_HANDOFF.md                                     |  43 --
 METHODOLOGY.md                                     |  51 --
 README.md                                          |  16 +-
 SCOPE.md                                           |  17 -
 config/lab_corr_policy.json                        |  17 +
 .../bcb/CRED_IMOB_PF_JUROS_MERCADO_25447.csv       | 358 ++++-----
 .../realestate/bcb/CRED_IMOB_PF_SALDO_20540.csv    | 454 +++++------
 data/raw/realestate/bcb/INCC_M_192.csv             | 628 ++++++++--------
 data/raw/realestate/bcb/SELIC_D_11.csv             |   4 +
 data/raw/realestate/manifest.json                  |  32 +-
 .../realestate/core/FipeZap_Aracaju_Total_core.csv |  98 +--
 ...alne\303\241rio_Cambori\303\272_Total_core.csv" | 194 ++---
 .../realestate/core/FipeZap_Barueri_Total_core.csv | 194 ++---
 .../core/FipeZap_Belo_Horizonte_Total_core.csv     | 404 +++++-----
 .../core/FipeZap_Bel\303\251m_Total_core.csv"      |  98 +--
 data/realestate/core/FipeZap_Betim_Total_core.csv  | 194 ++---
 .../core/FipeZap_Blumenau_Total_core.csv           | 194 ++---
 .../core/FipeZap_Bras\303\255lia_Total_core.csv"   | 372 ++++-----
 .../core/FipeZap_Campinas_Total_core.csv           | 316 ++++----
 .../core/FipeZap_Campo_Grande_Total_core.csv       | 194 ++---
 data/realestate/core/FipeZap_Canoas_Total_core.csv | 194 ++---
 .../core/FipeZap_Caxias_do_Sul_Total_core.csv      | 194 ++---
 .../core/FipeZap_Contagem_Total_core.csv           | 292 ++++----
 .../core/FipeZap_Cuiab\303\241_Total_core.csv"     |  98 +--
 .../core/FipeZap_Curitiba_Total_core.csv           | 328 ++++----
 .../realestate/core/FipeZap_Diadema_Total_core.csv | 194 ++---
 .../FipeZap_Florian\303\263polis_Total_core.csv"   | 328 ++++----
 .../core/FipeZap_Fortaleza_Total_core.csv          | 382 +++++-----
 .../core/FipeZap_Goi\303\242nia_Total_core.csv"    | 292 ++++----
 .../core/FipeZap_Guaruj\303\241_Total_core.csv"    | 316 ++++----
 .../core/FipeZap_Guarulhos_Total_core.csv          | 316 ++++----
 .../core/FipeZap_Itaja\303\255_Total_core.csv"     | 194 ++---
 .../realestate/core/FipeZap_Itapema_Total_core.csv | 194 ++---
 ..._Jaboat\303\243o_dos_Guararapes_Total_core.csv" | 194 ++---
 .../core/FipeZap_Joinville_Total_core.csv          | 194 ++---
 .../FipeZap_Jo\303\243o_Pessoa_Total_core.csv"     | 194 ++---
 .../core/FipeZap_Londrina_Total_core.csv           | 194 ++---
 .../core/FipeZap_Macei\303\263_Total_core.csv"     | 194 ++---
 data/realestate/core/FipeZap_Manaus_Total_core.csv | 194 ++---
 data/realestate/core/FipeZap_Natal_Total_core.csv  |  98 +--
 .../core/FipeZap_Niter\303\263i_Total_core.csv"    | 340 ++++-----
 .../core/FipeZap_Novo_Hamburgo_Total_core.csv      | 194 ++---
 data/realestate/core/FipeZap_Osasco_Total_core.csv | 316 ++++----
 .../realestate/core/FipeZap_Pelotas_Total_core.csv | 194 ++---
 .../core/FipeZap_Porto_Alegre_Total_core.csv       | 328 ++++----
 .../core/FipeZap_Praia_Grande_Total_core.csv       | 316 ++++----
 data/realestate/core/FipeZap_Recife_Total_core.csv | 376 +++++-----
 .../FipeZap_Ribeir\303\243o_Preto_Total_core.csv"  | 194 ++---
 .../core/FipeZap_Rio_de_Janeiro_Total_core.csv     | 434 +++++------
 .../core/FipeZap_Salvador_Total_core.csv           | 372 ++++-----
 .../core/FipeZap_Santa_Maria_Total_core.csv        | 194 ++---
 .../FipeZap_Santo_Andr\303\251_Total_core.csv"     | 340 ++++-----
 data/realestate/core/FipeZap_Santos_Total_core.csv | 316 ++++----
 ...ap_S\303\243o_Bernardo_do_Campo_Total_core.csv" | 340 ++++-----
 ...peZap_S\303\243o_Caetano_do_Sul_Total_core.csv" | 340 ++++-----
 .../FipeZap_S\303\243o_Jos\303\251_Total_core.csv" | 194 ++---
 ...3\243o_Jos\303\251_do_Rio_Preto_Total_core.csv" | 194 ++---
 ...303\243o_Jos\303\251_dos_Campos_Total_core.csv" | 194 ++---
 ...03\243o_Jos\303\251_dos_Pinhais_Total_core.csv" | 194 ++---
 .../FipeZap_S\303\243o_Leopoldo_Total_core.csv"    | 194 ++---
 .../FipeZap_S\303\243o_Lu\303\255s_Total_core.csv" |  98 +--
 .../core/FipeZap_S\303\243o_Paulo_Total_core.csv"  | 434 +++++------
 .../FipeZap_S\303\243o_Vicente_Total_core.csv"     | 316 ++++----
 .../core/FipeZap_Teresina_Total_core.csv           |  98 +--
 .../core/FipeZap_Vila_Velha_Total_core.csv         | 328 ++++----
 .../core/FipeZap_Vit\303\263ria_Total_core.csv"    | 328 ++++----
 .../FipeZap_\303\215ndice_FipeZAP_Total_core.csv"  | 434 +++++------
 docs/COMMIT_CHECKLIST_FINAL.md                     |   4 +-
 docs/INDEX.md                                      |  22 +-
 docs/OFERTA_COMERCIAL_MOTOR.md                     |  57 ++
 scripts/bench/run_graph_regime_universe.py         |   9 +-
 scripts/lab/build_local_finance_pack.py            |  19 +
 scripts/lab/run_corr_macro_offline.py              | 830 +++++++++++++++++++--
 scripts/ops/audit_frontend_payloads.py             |  13 +-
 scripts/ops/run_daily_jobs.cmd                     |   7 +-
 scripts/ops/run_daily_jobs.ps1                     |  27 +-
 website-ui/app/globals.css                         |  21 +
 website-ui/components/DashboardFilters.tsx         |   2 +-
 website-ui/components/MotorControlCenter.tsx       | 168 +++--
 website-ui/components/RegimeChart.tsx              |  27 +-
 website-ui/components/SectorDashboard.tsx          |  19 +-
 .../data/lab_corr_macro/latest/latest_release.json |   2 +-
 .../public/data/latest/prediction_truth_daily.json |  28 +-
 .../data/latest/prediction_truth_history.csv       |   1 +
 88 files changed, 9071 insertions(+), 8342 deletions(-)
```

### 4.2 Arquivos tracked com alteração

```text
AGENT_MEMORY.md
BUSINESS_OUTCOME.md
DEV_LOG.md
JOURNAL.md
MAC_HANDOFF.md
METHODOLOGY.md
README.md
SCOPE.md
config/lab_corr_policy.json
data/raw/realestate/bcb/CRED_IMOB_PF_JUROS_MERCADO_25447.csv
data/raw/realestate/bcb/CRED_IMOB_PF_SALDO_20540.csv
data/raw/realestate/bcb/INCC_M_192.csv
data/raw/realestate/bcb/SELIC_D_11.csv
data/raw/realestate/manifest.json
data/realestate/core/FipeZap_Aracaju_Total_core.csv
"data/realestate/core/FipeZap_Balne\303\241rio_Cambori\303\272_Total_core.csv"
data/realestate/core/FipeZap_Barueri_Total_core.csv
data/realestate/core/FipeZap_Belo_Horizonte_Total_core.csv
"data/realestate/core/FipeZap_Bel\303\251m_Total_core.csv"
data/realestate/core/FipeZap_Betim_Total_core.csv
data/realestate/core/FipeZap_Blumenau_Total_core.csv
"data/realestate/core/FipeZap_Bras\303\255lia_Total_core.csv"
data/realestate/core/FipeZap_Campinas_Total_core.csv
data/realestate/core/FipeZap_Campo_Grande_Total_core.csv
data/realestate/core/FipeZap_Canoas_Total_core.csv
data/realestate/core/FipeZap_Caxias_do_Sul_Total_core.csv
data/realestate/core/FipeZap_Contagem_Total_core.csv
"data/realestate/core/FipeZap_Cuiab\303\241_Total_core.csv"
data/realestate/core/FipeZap_Curitiba_Total_core.csv
data/realestate/core/FipeZap_Diadema_Total_core.csv
"data/realestate/core/FipeZap_Florian\303\263polis_Total_core.csv"
data/realestate/core/FipeZap_Fortaleza_Total_core.csv
"data/realestate/core/FipeZap_Goi\303\242nia_Total_core.csv"
"data/realestate/core/FipeZap_Guaruj\303\241_Total_core.csv"
data/realestate/core/FipeZap_Guarulhos_Total_core.csv
"data/realestate/core/FipeZap_Itaja\303\255_Total_core.csv"
data/realestate/core/FipeZap_Itapema_Total_core.csv
"data/realestate/core/FipeZap_Jaboat\303\243o_dos_Guararapes_Total_core.csv"
data/realestate/core/FipeZap_Joinville_Total_core.csv
"data/realestate/core/FipeZap_Jo\303\243o_Pessoa_Total_core.csv"
data/realestate/core/FipeZap_Londrina_Total_core.csv
"data/realestate/core/FipeZap_Macei\303\263_Total_core.csv"
data/realestate/core/FipeZap_Manaus_Total_core.csv
data/realestate/core/FipeZap_Natal_Total_core.csv
"data/realestate/core/FipeZap_Niter\303\263i_Total_core.csv"
data/realestate/core/FipeZap_Novo_Hamburgo_Total_core.csv
data/realestate/core/FipeZap_Osasco_Total_core.csv
data/realestate/core/FipeZap_Pelotas_Total_core.csv
data/realestate/core/FipeZap_Porto_Alegre_Total_core.csv
data/realestate/core/FipeZap_Praia_Grande_Total_core.csv
data/realestate/core/FipeZap_Recife_Total_core.csv
"data/realestate/core/FipeZap_Ribeir\303\243o_Preto_Total_core.csv"
data/realestate/core/FipeZap_Rio_de_Janeiro_Total_core.csv
data/realestate/core/FipeZap_Salvador_Total_core.csv
data/realestate/core/FipeZap_Santa_Maria_Total_core.csv
"data/realestate/core/FipeZap_Santo_Andr\303\251_Total_core.csv"
data/realestate/core/FipeZap_Santos_Total_core.csv
"data/realestate/core/FipeZap_S\303\243o_Bernardo_do_Campo_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Caetano_do_Sul_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Jos\303\251_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Jos\303\251_do_Rio_Preto_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Jos\303\251_dos_Campos_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Jos\303\251_dos_Pinhais_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Leopoldo_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Lu\303\255s_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Paulo_Total_core.csv"
"data/realestate/core/FipeZap_S\303\243o_Vicente_Total_core.csv"
data/realestate/core/FipeZap_Teresina_Total_core.csv
data/realestate/core/FipeZap_Vila_Velha_Total_core.csv
"data/realestate/core/FipeZap_Vit\303\263ria_Total_core.csv"
"data/realestate/core/FipeZap_\303\215ndice_FipeZAP_Total_core.csv"
docs/COMMIT_CHECKLIST_FINAL.md
docs/INDEX.md
docs/OFERTA_COMERCIAL_MOTOR.md
scripts/bench/run_graph_regime_universe.py
scripts/lab/build_local_finance_pack.py
scripts/lab/run_corr_macro_offline.py
scripts/ops/audit_frontend_payloads.py
scripts/ops/run_daily_jobs.cmd
scripts/ops/run_daily_jobs.ps1
website-ui/app/globals.css
website-ui/components/DashboardFilters.tsx
website-ui/components/MotorControlCenter.tsx
website-ui/components/RegimeChart.tsx
website-ui/components/SectorDashboard.tsx
website-ui/public/data/lab_corr_macro/latest/latest_release.json
website-ui/public/data/latest/prediction_truth_daily.json
website-ui/public/data/latest/prediction_truth_history.csv
```

### 4.3 Arquivos untracked

```text
.github/workflows/repo-healthcheck.yml
docs/INDEX_UNICO.md
docs/motor/FUNDAMENTACAO_TEORICA_ASSYNTRAX_LATEX.md
docs/motor/MANUAL_MESTRE_ASSYNTRAX.md
docs/motor/README.md
docs/motor/manual_300p/PLANO_EDITORIAL_300_PAGINAS.md
docs/motor/manual_300p/README.md
docs/motor/manual_300p/volumes/VOLUME_01.md
docs/motor/manual_300p/volumes/VOLUME_02.md
docs/motor/manual_300p/volumes/VOLUME_03.md
docs/motor/manual_300p/volumes/VOLUME_04.md
docs/motor/manual_300p/volumes/VOLUME_05.md
docs/motor/manual_300p/volumes/VOLUME_06.md
docs/motor/manual_300p/volumes/VOLUME_07.md
docs/motor/manual_300p/volumes/VOLUME_08.md
docs/motor/manual_300p/volumes/VOLUME_09.md
docs/motor/manual_300p/volumes/VOLUME_10.md
docs/motor/referencias/assyntrax_core.bib
docs/notes/AGENT_MEMORY.md
docs/notes/BUSINESS_OUTCOME.md
docs/notes/DEV_LOG.md
docs/notes/JOURNAL.md
docs/notes/MAC_HANDOFF.md
docs/notes/METHODOLOGY.md
docs/notes/README.md
docs/notes/SCOPE.md
docs/operacao/CHECKLIST_SITE_DEPLOY.md
docs/operacao/GITHUB_CANONICO.md
docs/operacao/GUIA_AUTONOMO_ASSYNTRAX.md
docs/operacao/HANDOFF_CONTEXTO_NOVO_CODEX_20260221.md
docs/operacao/NOMES_PASTAS_SAIDA.md
docs/operacao/README.md
docs/operacao/RELATORIO_VERIFICACAO_SITE_20260221.md
docs/operacao/REPO_HEALTHCHECK.md
docs/operacao/ROTINA_DIARIA_MASTER.md
docs/venda/DEMO_REUNIAO_GUIA.md
docs/venda/ESTUDO_DE_CASO_REAL_SETOR.md
docs/venda/PACOTES_ENTREGA_3_NIVEIS.md
docs/venda/PROPOSTA_CURTA.md
docs/venda/RELATORIO_EXECUTIVO_1_PAGINA.md
scripts/bench/consolidate_event_study_proof.py
scripts/bench/run_event_study_proof.py
scripts/bench/run_followup_123.py
scripts/bench/run_motor_470_diagnostics.py
scripts/bench/run_sector_and_crisis_suite.py
scripts/maintenance/archive_legacy_docs.py
scripts/maintenance/clean_old_results.py
scripts/maintenance/organize_workspace.sh
scripts/ops/install_launchd_daily_publish.sh
scripts/ops/launchd/com.assyntrax.daily-publish.plist
scripts/ops/publish_latest_if_gate_ok.py
scripts/ops/run_daily_jobs.sh
scripts/ops/run_daily_master.py
scripts/ops/run_daily_publish_site.sh
scripts/ops/run_repo_healthcheck.sh
scripts/ops/update_prediction_truth_daily.py
scripts/sync_lab_corr_to_website.sh
```

## 5) Estrutura técnica-chave para continuidade

### 5.1 Rotas do site (pages + APIs)

```text
website-ui/app/(site)/about/page.tsx
website-ui/app/(site)/contact/page.tsx
website-ui/app/(site)/methods/page.tsx
website-ui/app/(site)/page.tsx
website-ui/app/(site)/privacy/page.tsx
website-ui/app/(site)/product/page.tsx
website-ui/app/(site)/proposta/page.tsx
website-ui/app/(site)/pt/about/page.tsx
website-ui/app/(site)/pt/contact/page.tsx
website-ui/app/(site)/pt/methods/page.tsx
website-ui/app/(site)/pt/page.tsx
website-ui/app/(site)/pt/product/page.tsx
website-ui/app/(site)/pt/proposta/page.tsx
website-ui/app/api/assets/[asset]/route.ts
website-ui/app/api/assets/route.ts
website-ui/app/api/contact/route.ts
website-ui/app/api/dashboard/overview/route.ts
website-ui/app/api/figures/route.ts
website-ui/app/api/files/[...path]/route.ts
website-ui/app/api/graph/backtest/route.ts
website-ui/app/api/graph/hypertest/route.ts
website-ui/app/api/graph/official/route.ts
website-ui/app/api/graph/regimes-batch/route.ts
website-ui/app/api/graph/regimes/route.ts
website-ui/app/api/graph/sanity/route.ts
website-ui/app/api/graph/sector/route.ts
website-ui/app/api/graph/series-batch/route.ts
website-ui/app/api/graph/universe/route.ts
website-ui/app/api/graph/validation/route.ts
website-ui/app/api/history/route.ts
website-ui/app/api/index/route.ts
website-ui/app/api/lab/corr/latest/route.ts
website-ui/app/api/latest/route.ts
website-ui/app/api/methodology/route.ts
website-ui/app/api/ops/latest/route.ts
website-ui/app/api/overview/route.ts
website-ui/app/api/realestate/asset/route.ts
website-ui/app/api/realestate/series/route.ts
website-ui/app/api/realestate/summary/route.ts
website-ui/app/api/regime/route.ts
website-ui/app/api/risk-truth/route.ts
website-ui/app/api/run/latest/route.ts
website-ui/app/api/runs/route.ts
website-ui/app/api/search/route.ts
website-ui/app/api/sectors/alerts/route.ts
website-ui/app/app/aplicacoes/page.tsx
website-ui/app/app/casos/page.tsx
website-ui/app/app/dashboard/page.tsx
website-ui/app/app/finance/page.tsx
website-ui/app/app/imoveis/page.tsx
website-ui/app/app/macro/page.tsx
website-ui/app/app/metodologia/page.tsx
website-ui/app/app/operacao/page.tsx
website-ui/app/app/real-estate/page.tsx
website-ui/app/app/setores/page.tsx
website-ui/app/app/sobre/page.tsx
website-ui/app/app/teoria/page.tsx
website-ui/app/app/venda/page.tsx
```

### 5.2 Scripts operacionais críticos

```text
scripts/ops/__pycache__/audit_frontend_payloads.cpython-314.pyc
scripts/ops/__pycache__/build_daily_snapshot.cpython-314.pyc
scripts/ops/__pycache__/build_run_audit_pack.cpython-314.pyc
scripts/ops/__pycache__/daily_diff_report.cpython-314.pyc
scripts/ops/__pycache__/freeze_engine_snapshot.cpython-314.pyc
scripts/ops/__pycache__/log_codex_work.cpython-314.pyc
scripts/ops/__pycache__/manage_api_keys.cpython-314.pyc
scripts/ops/__pycache__/monitor_sector_alerts_drift.cpython-314.pyc
scripts/ops/__pycache__/publish_latest_if_gate_ok.cpython-314.pyc
scripts/ops/__pycache__/run_daily_master.cpython-314.pyc
scripts/ops/__pycache__/run_daily_pipeline.cpython-314.pyc
scripts/ops/__pycache__/run_daily_sector_alerts.cpython-314.pyc
scripts/ops/__pycache__/run_daily_validation.cpython-314.pyc
scripts/ops/__pycache__/run_dual_mode_compare.cpython-314.pyc
scripts/ops/__pycache__/run_monthly_revalidation.cpython-314.pyc
scripts/ops/__pycache__/safira_advisor.cpython-314.pyc
scripts/ops/__pycache__/update_prediction_truth_daily.cpython-314.pyc
scripts/ops/__pycache__/validate_output_contract.cpython-314.pyc
scripts/ops/audit_frontend_payloads.py
scripts/ops/build_daily_snapshot.py
scripts/ops/build_run_audit_pack.py
scripts/ops/cron_daily.sh
scripts/ops/cron_dual_mode_compare.sh
scripts/ops/cron_monthly_revalidation.sh
scripts/ops/daily_diff_report.py
scripts/ops/freeze_engine_snapshot.py
scripts/ops/install_launchd_daily_publish.sh
scripts/ops/install_launchd_sector_alerts.sh
scripts/ops/install_systemd_sector_alerts.sh
scripts/ops/launchd/com.assyntrax.daily-publish.plist
scripts/ops/launchd/com.assyntrax.sector-alerts.plist
scripts/ops/log_codex_work.py
scripts/ops/manage_api_keys.py
scripts/ops/monitor_sector_alerts_drift.py
scripts/ops/publish_latest_if_gate_ok.py
scripts/ops/register_tasks_windows.ps1
scripts/ops/run_daily_jobs.cmd
scripts/ops/run_daily_jobs.ps1
scripts/ops/run_daily_jobs.sh
scripts/ops/run_daily_master.py
scripts/ops/run_daily_pipeline.py
scripts/ops/run_daily_publish_site.sh
scripts/ops/run_daily_sector_alerts.py
scripts/ops/run_daily_validation.py
scripts/ops/run_dual_mode_compare.py
scripts/ops/run_monthly_revalidation.py
scripts/ops/run_repo_healthcheck.sh
scripts/ops/run_unified_pipeline.ps1
scripts/ops/safira_advisor.py
scripts/ops/systemd/assyntrax-sector-alerts.service
scripts/ops/systemd/assyntrax-sector-alerts.timer
scripts/ops/update_prediction_truth_daily.py
scripts/ops/validate_output_contract.py
```

### 5.3 Documentação de referência ativa

```text
docs/CODEX_WORKLOG.md
docs/COMMIT_CHECKLIST_FINAL.md
docs/CORE_PURE_MIGRATION_PLAN.md
docs/DAILY_PIPELINE.md
docs/DASHBOARD_SPEC.md
docs/ENGINE_FREEZE.md
docs/ENGINE_GUIDE.md
docs/GUIA_CONTINUIDADE_WINDOWS_E_MOTOR.txt
docs/INDEX.md
docs/INDEX_UNICO.md
docs/LEGACY_CANDIDATES.md
docs/MOTOR_ACTION_RULEBOOK.md
docs/MOTOR_PROVA_3P.md
docs/MOTOR_STATUS_OVERVIEW.md
docs/OFERTA_COMERCIAL_MOTOR.md
docs/OPS_EXECUTION_FLOW.md
docs/PACOTE_VENDA_CHECKLIST.md
docs/PILOTO_30D_PLAYBOOK.md
docs/PRINCIPLES.md
docs/PRODUCT_THESIS.md
docs/REAL_ESTATE_ARCH.md
docs/REGIME_METHODS_STUDY.md
docs/REGRESSION_CHECKLIST_PRECOMMIT.md
docs/REPO_REFACTOR_PLAN.md
docs/SECTOR_ALERTS_API_CONTRACT.md
docs/commit_prep_inventory.csv
docs/graph_engine_deps.md
docs/graph_engine_frontend_contract.md
docs/graph_engine_plan.md
docs/graph_engine_universe_40.md
docs/motor/FUNDAMENTACAO_TEORICA_ASSYNTRAX_LATEX.md
docs/motor/LIVRO_MOTOR_ASSYNTRAX_300P.md
docs/motor/MANUAL_MESTRE_ASSYNTRAX.md
docs/motor/README.md
docs/motor/manual_300p/PLANO_EDITORIAL_300_PAGINAS.md
docs/motor/manual_300p/README.md
docs/motor/referencias/assyntrax_core.bib
docs/motor_dependency_matrix.csv
docs/notes/AGENT_MEMORY.md
docs/notes/BUSINESS_OUTCOME.md
docs/notes/DEV_LOG.md
docs/notes/JOURNAL.md
docs/notes/MAC_HANDOFF.md
docs/notes/METHODOLOGY.md
docs/notes/README.md
docs/notes/SCOPE.md
docs/operacao/CHECKLIST_SITE_DEPLOY.md
docs/operacao/GITHUB_CANONICO.md
docs/operacao/GUIA_AUTONOMO_ASSYNTRAX.md
docs/operacao/HANDOFF_CONTEXTO_NOVO_CODEX_20260221.md
docs/operacao/NOMES_PASTAS_SAIDA.md
docs/operacao/README.md
docs/operacao/RELATORIO_VERIFICACAO_SITE_20260221.md
docs/operacao/REPO_HEALTHCHECK.md
docs/operacao/ROTINA_DIARIA_MASTER.md
docs/ops_playbook.md
docs/venda/DEMO_REUNIAO_GUIA.md
docs/venda/ESTUDO_DE_CASO_REAL_SETOR.md
docs/venda/PACOTES_ENTREGA_3_NIVEIS.md
docs/venda/PROPOSTA_CURTA.md
docs/venda/RELATORIO_EXECUTIVO_1_PAGINA.md
```

## 6) Status de deploy e verificação

- Relatório de verificação completo: `docs/operacao/RELATORIO_VERIFICACAO_SITE_20260221.md`.
- Checklist de deploy: `docs/operacao/CHECKLIST_SITE_DEPLOY.md`.
- Governança GitHub canônico: `docs/operacao/GITHUB_CANONICO.md`.

## 7) Riscos e pendências técnicas abertas

1. Worktree está **bem sujo** (muitos dados/docs/scripts em alteração). Não commitar em bloco.
2. Há muitos arquivos de dados imobiliários atualizados localmente; separar commit de dados vs commit de código.
3. Manter política de `pull --ff-only` e healthcheck antes de push para evitar divergência Mac/Windows.
4. Alguns artefatos de métricas ainda vêm com qualidade/recall parcial em certos recortes; continuar validação setorial/event study.

## 8) Comandos oficiais de operação (copiar e usar)

```bash
# 1) Sincronizar com remoto canônico
git fetch origin --prune
git pull --ff-only origin main

# 2) Healthcheck completo
./scripts/ops/run_repo_healthcheck.sh

# 3) Pipeline diário master
python3 scripts/ops/run_daily_master.py

# 4) Publicar snapshot apenas se gate permitir
python3 scripts/ops/publish_latest_if_gate_ok.py

# 5) Frontend local
cd website-ui
npm run build
npm run dev
```

## 9) Contexto pronto para colar no novo Codex

```text
Projeto: Assyntrax (motor de detecção de regimes estruturais em séries financeiras + site operacional Next.js).

Objetivo atual:
- Continuar evolução do motor e do site sem perder governança entre máquinas (Windows/Mac) e mantendo GitHub (origin/main) como fonte canônica.

Estado técnico atual:
- Motor principal: scripts/lab/run_corr_macro_offline.py
- Política oficial: config/lab_corr_policy.json
- Pipeline ops diário: scripts/ops/run_daily_master.py
- Gate de publicação + bloqueio: publish_gate.json + PUBLISH_BLOCKED
- Site/API: website-ui/app + website-ui/app/api
- Documentação mestre: docs/motor/MANUAL_MESTRE_ASSYNTRAX.md
- Verificação de site/deploy: docs/operacao/RELATORIO_VERIFICACAO_SITE_20260221.md

Regras de trabalho obrigatórias:
1) Sempre sincronizar antes de editar:
   git fetch origin --prune
   git pull --ff-only origin main
2) Rodar healthcheck antes de push:
   ./scripts/ops/run_repo_healthcheck.sh
3) Commits pequenos por escopo (não misturar dados, código e docs no mesmo commit sem necessidade).
4) Nunca usar reset destrutivo para “sincronizar” ambiente compartilhado.

Pontos recentes já implementados:
- Correções de estados sem dados/falha de carregamento em componentes críticos.
- Inclusão de rota /app/venda e eliminação de dead-end.
- Página de teoria com fórmulas/equações e referências.
- Correção de português e blindagem contra links/cards com URL vazia.

O que preciso que você faça primeiro:
1) Ler docs/operacao/HANDOFF_CONTEXTO_NOVO_CODEX_20260221.md.
2) Verificar git status e separar escopo do próximo commit.
3) Rodar healthcheck e build do site.
4) Só então implementar próximos ajustes priorizados.
```

## 10) Observação final

Este arquivo foi criado para continuidade imediata no novo Codex, preservando histórico, estado local e playbook operacional em um único ponto de entrada.
