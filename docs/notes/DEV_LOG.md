# Dev Log (Resumo Atual)

## 2026-02
- API de engine unificada em `engine/`.
- Imports de scripts migrados para `engine.*`.
- Wrappers criados em `spa/` e `graph_engine/` para compatibilidade.
- Pipeline diario consolidado em `scripts/ops/`.
- Contrato e gates de saida formalizados via config.
- Frontend ajustado para leitura de snapshots validados.
- Normalizacao defensiva de `domain` na API do site para evitar valores sujos/typos no payload.
- Parser JSONL robusto no backend do site para tolerar `NaN/Infinity` sem derrubar rotas.
- Diagnostico de ativos inconclusivos automatizado em `results/validation/inconclusive_diagnosis_latest.csv`.
- Suite de validacao de produto executada com status `ok` (18/18) em `scripts/bench/validation/run_product_pipeline.py`.
- VERDICT consolidado com status `pass` em `results/validation/VERDICT.json`.

## Fechamento do dia (motor + site + deploy)
- Contrato operacional mantido: `status`, `regime`, `confidence`, `quality`, `instability_score`, `reason`, `data_adequacy`, `run_id`.
- Snapshot validado mantido como fonte principal da API/site.
- Checklist tecnico executado:
  - `npm run lint` (ok)
  - `npm run typecheck` (ok)
  - `npm run build` (bloqueio de ambiente: `spawn EPERM` em Windows, pendente de saneamento local).
- Proxima acao direta antes do deploy final:
  - sanear ambiente Windows para build de producao do Next e rerodar smoke completo de rotas/API.

## Proximo checkpoint
- Limpeza final de legado nao operacional.
- Atualizacao de docs para rename do repositorio.
- Checklist final de regressao pre-commit.

## 2026-02-20 16:55 UTC - Motor núcleo + 470 ativos (diagnóstico consolidado)

### Implementação
- Criado e evoluído `scripts/bench/run_motor_470_diagnostics.py`.
- Incluído no script:
  - revisão de regras do classificador;
  - tuning de pesos/limiares/histerese;
  - modos `conservative` e `aggressive` com comparação objetiva;
  - política final congelada em JSON;
  - estabilidade do vetor principal (overlap);
  - teste formal de significância por janela;
  - validação de cobertura e consistência dos 470 ativos;
  - rankings de ativos sensíveis/estáveis;
  - detecção de weak signal e fila de revisão.

### Run mais recente
- pasta: `results/motor_470_program/20260220T165515Z`
- motor (L10):
  - baseline: recall=0.8333 | precision=0.1290 | falso_alarme/ano=3.8813
  - conservador: recall=0.8333 | precision=0.1429 | falso_alarme/ano=3.4501
  - agressivo: recall=1.0000 | precision=0.1633 | falso_alarme/ano=5.8939
- modo recomendado: `conservative` (melhor custo-benefício para produção)

### Universo 470
- cobertura: 470/470 com arquivo de regime
- weak signal: 123 ativos
- issues de série (gap/duplicata/curta/conf fora/estado inválido): 0

### Arquivos-chave gerados
- `results/motor_470_program/20260220T165515Z/motor_policy_final.json`
- `results/motor_470_program/20260220T165515Z/motor_mode_comparison.csv`
- `results/motor_470_program/20260220T165515Z/motor_significance_by_window.csv`
- `results/motor_470_program/20260220T165515Z/universe_asset_diagnostics_enriched.csv`
- `results/motor_470_program/20260220T165515Z/universe_top_sensitive.csv`
- `results/motor_470_program/20260220T165515Z/universe_top_stable.csv`
- `results/motor_470_program/20260220T165515Z/universe_weak_signal_assets.csv`
- `results/motor_470_program/20260220T165515Z/universe_series_consistency.csv`
- `results/motor_470_program/20260220T165515Z/universe_review_queue.csv`

## 2026-02-20 17:01 UTC - Follow-up 1,2,3 executado

### Escopo executado
1) teste continuo da politica conservadora
2) triagem dos 123 ativos de sinal fraco
3) diagnostico final por setor (foco comercial)

### Script novo
- `scripts/bench/run_followup_123.py`

### Saida principal
- `results/followup_123/20260220T170102Z/summary.json`

### Pontos principais
- Politica conservadora (l10): recall=0.8333, precision=0.1429, falso_alerta=3.4501/ano, antecedencia=7.8
- Cobertura macro valida: 1753/1989 dias (ultimo dia valido 2025-03-04)
- Weak signal total: 123
- Sem problema de dados por ativo (0 com gap/duplicata/faixa invalida)
- Setores principais mais fragilizados: energy, real_estate, consumer_discretionary, utilities, consumer_staples

### Documentacao comercial atualizada
- `docs/OFERTA_COMERCIAL_MOTOR.md` com secao "Atualizacao pratica (1, 2 e 3 executados)"

## 2026-02-20 17:12 UTC - Setores + crises historicas (suite completa)

### Script novo
- `scripts/bench/run_sector_and_crisis_suite.py`

### Ultimo run
- `results/sector_crisis_suite/20260220T171218Z`

### Setores
- auditoria de mapeamento: 470 ativos, 14 setores, 0 vazios, 0 nomes corrigidos
- reprocessado risco/confianca/%instaveis + mudanca semanal de estado por setor
- ranking principal (>=10 ativos): energy, real_estate, financials, utilities, consumer_discretionary
- plano de acao por nivel salvo em `sector_action_plan.txt`

### Crises historicas (janelas 1/5/10/20)
- eventos: retorno extremo (ret_tail) e queda acumulada (drawdown20)
- comparacoes: baseline volatilidade, baseline retorno e baseline aleatorio
- resultado: sinal operacional de antecipacao, sem evidencia estatistica forte (p<0.05 nao confirmado)
- metrica forte em drawdown20 L10: recall 0.727 com falso alerta 14.684/ano, p_random=0.051

### Arquivos principais
- `results/sector_crisis_suite/20260220T171218Z/diagnostics_sectors_reprocessed.csv`
- `results/sector_crisis_suite/20260220T171218Z/sector_weekly_state_change.csv`
- `results/sector_crisis_suite/20260220T171218Z/sector_action_ranking_main.csv`
- `results/sector_crisis_suite/20260220T171218Z/report_crisis_antecipa_ou_nao.txt`
- `results/sector_crisis_suite/20260220T171218Z/crisis_event_metrics_summary.csv`

## 2026-02-20 17:28 UTC - Site/Painel + Operacao + Produto + Organizacao

### Painel e site
- API setorial ampliada: `website-ui/app/api/sectors/alerts/route.ts`
  - filtros via query (`days`, `level`, `sector`, `min_assets`)
  - `summary_simple`, `data_quality`, `limits`
- Painel setorial atualizado: `website-ui/components/SectorAlertsDashboard.tsx`
  - resumo simples do estado atual
  - filtros extras
  - bloco de qualidade de dados
  - bloco de limites do motor
  - explicacao humana do alerta com destaque
- Dashboard principal: `website-ui/app/app/dashboard/page.tsx`
  - estado simples do motor no topo
- Dashboard de ativos: `website-ui/components/DashboardSimple.tsx`
  - filtros de ativo, setor, janela (daily/weekly) e periodo (90/180/365/730)
- Site comercial:
  - `website-ui/app/(site)/proposta/page.tsx`
  - `website-ui/app/(site)/pt/proposta/page.tsx`
  - `website-ui/app/(site)/en/proposal/page.tsx`
  - menu com link de proposta em `website-ui/components/SiteHeader.tsx`

### Operacao e rotina
- Novo orquestrador unico: `scripts/ops/run_daily_master.py`
  - rotina diaria consolidada
  - checagem de sanidade
  - gate de publicacao (`publish_gate.json`)
  - bloqueio com `PUBLISH_BLOCKED` quando necessario
  - relatorio diario em texto
  - comparacao dia a dia
  - padrao de saida em `results/ops/runs/<run_id>`
- Wrappers Windows atualizados:
  - `scripts/ops/run_daily_jobs.ps1`
  - `scripts/ops/run_daily_jobs.cmd`

### Produto e venda
- Criados materiais:
  - `docs/venda/PROPOSTA_CURTA.md`
  - `docs/venda/PACOTES_ENTREGA_3_NIVEIS.md`
  - `docs/venda/RELATORIO_EXECUTIVO_1_PAGINA.md`
  - `docs/venda/ESTUDO_DE_CASO_REAL_SETOR.md`
  - `docs/venda/DEMO_REUNIAO_GUIA.md`
- Oferta conectada ao kit comercial: `docs/OFERTA_COMERCIAL_MOTOR.md`

### Organizacao
- Estrutura por tema:
  - `docs/motor/README.md`
  - `docs/operacao/README.md`
  - `docs/operacao/ROTINA_DIARIA_MASTER.md`
  - `docs/operacao/NOMES_PASTAS_SAIDA.md`
  - `docs/INDEX_UNICO.md`
- Arquivo de legado:
  - `scripts/maintenance/archive_legacy_docs.py` (executado com `--apply`, removido no cleanup de 2026-02-27)
  - copias antigas em `docs/historico/arquivo_20260220/` (arquivo removido no cleanup de 2026-02-27)
- Limpeza de resultados antigos:
  - `scripts/maintenance/clean_old_results.py` (dry-run pronto)

### Validacao
- Python compile: ok para scripts novos/alterados.
- Frontend: `npm run -s typecheck` ok, `npm run -s lint` ok.

## 2026-02-20 18:48 UTC - Frontes 1 e 2 completos

### Frente 1 (dados e cobertura)
- Causa raiz encontrada: pack antigo de 41 ativos + calendario com datas esparsas (feriados de bolsa com apenas cripto).
- Ajustes:
  - `scripts/lab/build_local_finance_pack.py` ganhou `--min-date-coverage` para remover datas com baixa cobertura.
  - `scripts/lab/run_corr_macro_offline.py` robustez ajustada para janelas ausentes sem quebrar.
- Novo pack:
  - `results/finance_download/local_pack_20260220T173837Z`
  - 470 ativos, 0 faltantes, periodo ate 2026-02-12
- Run macro final promovido:
  - `results/lab_corr_macro/20260220T174540Z`
  - T60: 1981/1981 valido ate 2026-02-12
  - T120: 1921/1921 valido ate 2026-02-12
  - T252: 1789/1789 valido ate 2026-02-12
  - deployment_gate: desbloqueado
  - latest_release atualizado para esse run

### Frente 2 (prova tecnica de antecipacao)
- Novos scripts:
  - `scripts/bench/run_event_study_proof.py` (orquestrador de prova reforcada)
  - `scripts/bench/consolidate_event_study_proof.py` (consolidacao final multi-politica)
- Rodadas event study executadas (1,5,10,20 dias) com 5 politicas:
  - regime_entry, regime_guarded, regime_balanced, score_q80, score_q90
  - raws em `results/event_study_proof_raw/*`
- Consolidado final:
  - `results/event_study_proof/20260220T184728Z`
  - melhor politica: `regime_entry`
  - veredito: "Sem evidencia estatistica forte; sinal operacional presente"
- Destaque:
  - regime_entry tem recall alto (drawdown L10=0.727, L20=0.909),
  - mas p_vs_random ainda nao cruza <0.05 de forma robusta no criterio principal.
  - contra baselines simples por evento, McNemar mostrou vantagem estatistica (p baixo), mas baseline aleatorio segue o limitante para claim forte.

## 2026-02-22 UTC - checkpoint de migracao (Windows -> Mac)
- commit consolidado com copiloto/plataforma/Model C GNN: `e36b02fd745b283f9ebacbc9d012e12009b5f8cf`
- commit de contexto operacional: `85dc68bb2e911779bb88ca9be263167a96950778`
- model C em `scripts/ops/build_copilot_shadow.py` usa `gnn_checkpoint` quando existe `models/model_c_gnn_checkpoint.json`
- contrato/gate do copiloto mantido (mesmo payload para site/API, com fallback controlado para `shadow_proxy`)
- regra operacional reforcada: informar ETA antes de rotinas longas
