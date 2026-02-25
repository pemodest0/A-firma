# Livro do Eigen Engine (Plano de 300 Paginas)

## Objetivo
Este arquivo e o guia mestre para consolidar todo o conhecimento do Eigen Engine em um livro tecnico longo, auditavel e util para produto, operacao e venda institucional.

Meta editorial: 300 paginas.

## Como usar este guia
1. Cada bloco abaixo tem meta de paginas e entregaveis.
2. Escreva primeiro o conteudo que ja existe em codigo e artefatos.
3. So depois inclua secoes de pesquisa externa.
4. Mantenha links para arquivos reais do repositorio em cada capitulo.

## Estrutura sugerida (15 blocos x 20 paginas = 300)

### Bloco 1 (20p) - Fundamentos do problema
- O que o Eigen Engine faz e o que nao faz.
- Definicao de regime: estavel, transicao, estresse, dispersao.
- Fronteiras de validade e limites matematicos.

Fontes internas:
- `scripts/lab/run_corr_macro_offline.py`
- `config/lab_corr_policy.json`
- `website-ui/public/data/lab_corr_macro/latest/summary.json`

### Bloco 2 (20p) - Dados e universo de ativos
- Universo fixo (470+ ativos), cobertura, faltas e filtros.
- Janela temporal e regras de warmup.
- Politica de qualidade de dados.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/asset_regime_diagnostics.csv`
- `website-ui/public/data/lab_corr_macro/latest/asset_sector_summary.json`
- `website-ui/public/data/lab_corr_macro/latest/qa_checks.json`

### Bloco 3 (20p) - Nucleo espectral
- Matriz de correlacao rolling.
- Autovalores, p1, deff, entropia espectral.
- Interpretacao economica das metricas.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/macro_timeseries_T60.csv`
- `website-ui/public/data/lab_corr_macro/latest/macro_timeseries_T120.csv`
- `website-ui/public/data/lab_corr_macro/latest/macro_timeseries_T252.csv`

### Bloco 4 (20p) - Classificador de regime
- Regras matematicas do classificador.
- Histerese e permanencia minima.
- Comparacao entre modo conservador e agressivo.

Fontes internas:
- `config/lab_corr_policy.json`
- `website-ui/public/data/lab_corr_macro/latest/regime_series_T120.csv`
- `website-ui/public/data/lab_corr_macro/latest/alert_levels_T120.csv`

### Bloco 5 (20p) - Causalidade e integridade temporal
- Sem look-ahead: principios e implementacao.
- Calibracao walk-forward.
- Evidencias de causalidade no pipeline.

Fontes internas:
- `scripts/lab/run_corr_macro_offline.py`
- `website-ui/app/api/lab/corr/latest/route.ts`
- `website-ui/public/data/lab_corr_macro/latest/latest_release.json`

### Bloco 6 (20p) - Significancia e ruido
- Shuffle e block bootstrap.
- Intervalos de confianca.
- p-value por janela e interpretacao.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/significance_summary_by_window.csv`
- `website-ui/public/data/lab_corr_macro/latest/era_evaluation_T120.json`

### Bloco 7 (20p) - Robustez
- Subamostragem de ativos.
- Sensibilidade de parametros.
- Consistencia 60/120/252.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/summary_compact.txt`
- `website-ui/public/data/lab_corr_macro/latest/summary.json`

### Bloco 8 (20p) - Diagnostico por ativo
- Ranking de risco e confianca.
- Troca de regime por ativo (30/90/180 dias).
- Casos de sinal forte vs sinal fraco.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/asset_regime_diagnostics.csv`

### Bloco 9 (20p) - Diagnostico setorial
- Risco medio por setor.
- Percentual de instaveis.
- Mudanca semanal e ranking setorial.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/sector_regime_diagnostics.csv`
- `website-ui/public/data/sectors/latest/sector_alert_levels_latest.csv`
- `website-ui/public/data/sectors/latest/weekly_compare.json`

### Bloco 10 (20p) - Estudos de eventos e crises
- Metodologia de evento automatico.
- Janela 1/5/10/20 dias.
- Recall, precisao, falso alarme e lead time.

Fontes internas:
- `website-ui/public/data/lab_corr_macro/latest/case_studies_T120.csv`
- `website-ui/public/data/sectors/latest/sector_rank_l5.csv`
- `website-ui/public/data/sectors/latest/report_sector_event_study.txt`

### Bloco 11 (20p) - Operacao diaria
- Rotina diaria unica.
- Gates de publicacao.
- O que bloqueia release e por que.

Fontes internas:
- `scripts/ops/run_daily_master.py`
- `scripts/ops/publish_latest_if_gate_ok.py`
- `scripts/ops/update_prediction_truth_daily.py`

### Bloco 12 (20p) - API, dashboard e rastreabilidade
- Contratos de API.
- Como o app consome os snapshots.
- Post-mortem de alertas (explicabilidade).

Fontes internas:
- `website-ui/app/api/lab/corr/latest/route.ts`
- `website-ui/app/api/sectors/alerts/route.ts`
- `website-ui/components/MotorControlCenter.tsx`
- `website-ui/components/SectorAlertsDashboard.tsx`

### Bloco 13 (20p) - Produto e comercializacao
- Posicionamento correto: monitor de risco.
- O que e vendavel hoje.
- Limites e linguagem regulatoria.

Fontes internas:
- `docs/OFERTA_COMERCIAL_MOTOR.md`
- `website-ui/app/app/venda/page.tsx`

### Bloco 14 (20p) - Governanca tecnica
- Versionamento de politica.
- Release management e rollback.
- Controle entre ambientes (Windows, Mac, CI).

Fontes internas:
- `config/lab_corr_policy.json`
- `website-ui/public/data/lab_corr_macro/latest/latest_release.json`
- `README.md`

### Bloco 15 (20p) - Roadmap 12 meses
- Melhorias de curto, medio e longo prazo.
- Hipoteses para pesquisa futura.
- Plano de validacao continua.

Fontes internas:
- `docs/INDEX.md`
- `docs/INDEX_UNICO.md`

## Tabela de rastreabilidade (obrigatoria no livro final)
Para cada capitulo, incluir:
1. Pergunta principal.
2. Formula/metrica usada.
3. Arquivo de origem.
4. Data de extracao.
5. Resultado principal.
6. Limite conhecido.

## Padrao de escrita (recomendado)
1. Linguagem simples.
2. Sem promessa de retorno.
3. Sempre separar: fato medido, inferencia, limite.
4. Sempre informar data absoluta do dado.

## Checklist de fechamento do livro
1. Todos os capitulos com fonte interna citada.
2. Todas as tabelas com periodo e unidade.
3. Todas as figuras com legenda e interpretacao.
4. Secao unica de riscos e limites.
5. Anexo com contratos de API e estrutura de pastas.

## Anexo A - Arquivos vitais do motor (snapshot atual)
- `website-ui/public/data/lab_corr_macro/latest/summary.json`
- `website-ui/public/data/lab_corr_macro/latest/alert_levels_T120.csv`
- `website-ui/public/data/lab_corr_macro/latest/asset_regime_diagnostics.csv`
- `website-ui/public/data/lab_corr_macro/latest/sector_regime_diagnostics.csv`
- `website-ui/public/data/sectors/latest/sector_alert_levels_latest.csv`
- `website-ui/public/data/sectors/latest/sector_rank_l5.csv`

## Anexo B - Regra de ouro de sincronizacao multi-maquina
Github `origin/main` e a fonte unica de verdade.

Fluxo minimo diario:
1. `git fetch origin`
2. `git checkout main`
3. `git pull --rebase origin main`
4. Trabalhar e testar local.
5. `git add ... && git commit ...`
6. `git push origin main`

Se um ambiente ficou muito atrasado, sincronizar com:
`git fetch origin && git checkout main && git reset --hard origin/main`

(Use apenas quando quiser descartar alteracoes locais nesse ambiente.)
