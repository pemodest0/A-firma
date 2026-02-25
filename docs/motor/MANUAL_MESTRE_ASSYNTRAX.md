# Manual Mestre do Eigen Engine (Assyntrax)

Versao: v1.1 (nomenclatura oficial Assyntrax + Eigen Engine)
Data: 2026-02-25
Status: ativo

---

## Nomenclatura oficial (produto)

- `Assyntrax`: empresa, marca e plataforma.
- `Eigen Engine`: motor estrutural oficial da plataforma.

Compatibilidade: caminhos legados de arquivo podem manter `ASSYNTRAX` no nome para preservar historico e links internos.

---

## 1. Objetivo do manual

Este documento consolida, em um unico lugar, como o Eigen Engine funciona de ponta a ponta:

- como os dados entram,
- como as métricas são calculadas,
- como os regimes são definidos,
- como os alertas são gerados,
- como o gate de publicação decide liberar ou bloquear,
- e como chegamos às conclusões operacionais.

Este manual não é material de marketing. É documento técnico-operacional para auditoria, manutenção e continuidade.

---

## 1.1 Mapa de arquivos e funcoes

Inventario tecnico vivo (arquivo por arquivo + funcao por funcao):

- `docs/motor/EIGEN_ENGINE_FILE_FUNCTION_MAP.md`

Regra: sempre que o motor ganhar/remover funcao no fluxo oficial, esse mapa deve ser atualizado no mesmo ciclo de commit.

---

## 2. O que o Eigen Engine e (e o que nao e)

### 2.1 O que é

O Eigen Engine e um sistema de diagnostico estrutural de mercado baseado em correlacao rolling e analise espectral.

Ele mede:

- concentração de variância no primeiro modo (`p1`),
- dimensão efetiva (`deff`),
- instabilidade de estrutura no tempo (via mudanças de `p1`, `deff` e overlap de autovetor principal),
- robustez entre janelas,
- consistência estatística frente a baseline de ruído.

### 2.2 O que não é

- Não prevê preço alvo de ativo.
- Não prevê data exata de crash.
- Não garante retorno financeiro.
- Não substitui decisão humana.

---

## 3. Fontes e componentes oficiais

## 3.1 Script núcleo

- `scripts/lab/run_corr_macro_offline.py`

Responsável por:

- cálculo das séries espectrais por janela,
- classificação de regime,
- backtest defensivo,
- alertas operacionais,
- significância,
- diagnósticos por ativo e setor,
- QA,
- deployment gate,
- release pointer.

## 3.2 Política oficial

- `config/lab_corr_policy.json`

Contém parâmetros oficiais de:

- dados,
- classificação,
- exposição,
- backtest,
- gate,
- alertas,
- calibração,
- casos.

## 3.3 API de consumo no app

- `website-ui/app/api/lab/corr/latest/route.ts`

Expondo contrato consolidado para UI.

---

## 4. Política oficial atual (resumo técnico)

Fonte: `config/lab_corr_policy.json`

## 4.1 Dados

- período padrão: `2018-01-01` a `2026-02-12`
- cobertura núcleo: `0.95`
- cobertura por janela: `0.98`
- `min_assets`: `25`
- janela oficial: `120`
- dias úteis: `true`

## 4.2 Regime

- histerese: `3 dias`
- modo de limiar: `walk_forward`
- histórico mínimo para calibração causal: `252 dias`
- pesos de transição:
  - `dp1`: `0.45`
  - `ddeff`: `0.45`
  - `overlap_instability`: `0.10`
- mapa de exposição:
  - stress: `0.10`
  - transition: `0.40`
  - stable: `0.70`
  - dispersion: `0.90`

## 4.3 Gate

- `min_joint_majority_60d`: `0.30`
- `max_abs_delta_p1`: `0.01`
- `max_abs_delta_deff`: `1.0`
- `max_active_cluster_alerts`: `1`
- `max_insufficient_ratio`: `0.05`
- `min_n_used_ratio`: `0.90`
- `q_min`: `0.12`

## 4.4 Alertas

- risco amarelo: quantil `0.70`
- risco vermelho: quantil `0.90`
- confiança mínima amarelo: `0.35`
- confiança mínima vermelho: `0.45`
- persistência de nível: `2 em 3`

---

## 5. Fluxo completo de análise

## 5.1 Etapas

1. Carrega universo e retornos.
2. Filtra cobertura por janela e por ativo.
3. Para cada janela (`T60`, `T120`, `T252`), calcula:
   - correlação,
   - espectro,
   - métricas estruturais,
   - baseline de ruído (shuffle + block bootstrap).
4. Calcula robustez temporal entre janelas.
5. Classifica regime oficial na janela T120.
6. Calcula backtest de exposição por regime.
7. Gera alertas operacionais e níveis.
8. Gera significância estatística por janela.
9. Gera diagnóstico por ativo e setor.
10. Executa QA e gate de deployment.
11. Se gate aprovado, atualiza release pointer.

## 5.2 Artefatos

Arquivos principais gerados no run:

- `macro_timeseries_T120.csv`
- `regime_series_T120.csv`
- `backtest_T120.csv`
- `backtest_summary_T120.json`
- `operational_alerts_T120.json`
- `alert_levels_T120.csv`
- `significance_summary_by_window.csv`
- `asset_regime_diagnostics.csv`
- `sector_regime_diagnostics.csv`
- `deployment_gate.json`
- `summary.json`
- `summary_compact.txt`
- `latest_release.json`

---

## 6. Matemática do núcleo

## 6.1 Métricas espectrais

Para cada data e janela:

- monta matriz de correlação `C_t` dos retornos válidos,
- extrai autovalores `lambda_1 ... lambda_N`,
- calcula:
  - `p1`: fração da variância explicada por `lambda_1`,
  - `deff`: dimensão efetiva,
  - `top5`: concentração dos 5 maiores autovalores,
  - `entropy` e estrutura de cluster (quando disponível).

## 6.2 Instabilidade do modo principal

O script calcula overlap do autovetor principal quando aplicável:

- `eigvec_overlap_1d`
- `eigvec_instability_1d = 1 - eigvec_overlap_1d`

Isso separa:

- concentração estável de correlação,
- de mudança real de direção estrutural.

## 6.3 Score de transição

Definição no classificador:

`transition_score = w_dp1 * z(|dp1_5|) + w_ddeff * z(|ddeff_5|) + w_overlap * z(overlap_instability)`

Com pesos oficiais de política:

- `w_dp1 = 0.45`
- `w_ddeff = 0.45`
- `w_overlap = 0.10`

## 6.4 Limiar causal (walk-forward)

No modo oficial:

- limiares em `t` são calibrados com histórico até `t-1`,
- sem uso de futuro,
- warmup mínimo definido por `walkforward_min_history_days`.

Limiar usado:

- `q20` e `q80` para `p1` e `deff`,
- `q80` para `|dp1_5|`, `|ddeff_5|` e `transition_score`.

## 6.5 Regra de estado bruto

Por dia:

- `stress` se `p1 >= q80(p1)` e `deff <= q20(deff)`
- `dispersion` se `p1 <= q20(p1)` e `deff >= q80(deff)`
- `transition` se qualquer condição de transição excede limiar q80
- caso contrário: `stable`

## 6.6 Histerese

Aplica persistência mínima (`hysteresis_days`) antes de aceitar troca de estado.

Objetivo:

- reduzir flicker,
- evitar troca espúria diária.

---

## 7. Como chegamos às conclusões de regime

Esta é a parte central de interpretação.

## 7.1 Modo de análise que usamos

As conclusoes nao vem de uma unica metrica. O Eigen Engine combina quatro blocos:

1. Estrutura agregada (`p1`, `deff`, espectro)
2. Dinâmica de mudança (`dp1_5`, `ddeff_5`, overlap)
3. Robustez temporal (consistência entre janelas)
4. Significância contra ruído (bootstrap)

## 7.2 Regra prática de leitura

- Se concentração sobe (`p1`) e dimensão efetiva cai (`deff`), estrutura tende a stress.
- Se transição acelera sem consolidar thresholds, estado tende a transição.
- Se distribuição fica menos concentrada e deff sobe, tende a dispersão.
- Se nada dispara, permanece estável.

## 7.3 Critério de confiança do resultado

A confiança operacional cresce quando:

- há consistência entre janelas,
- sinais resistem ao baseline de ruído,
- QA não acusa insuficiência estrutural,
- gate de publicação permanece liberado.

## 7.4 Como evitamos overfitting na conclusão

- thresholds causais (`walk_forward`),
- validação com baseline e bootstrap,
- gate com limites explícitos,
- políticas versionadas via `lab_corr_policy.json`.

---

## 8. Robustez e significância

## 8.1 Baseline de ruído

Em `_process_window`, o script calcula:

- shuffle baseline
- block bootstrap baseline

Métricas derivadas:

- `p1_shuffle`, `deff_shuffle`
- `p1_bootstrap`, `deff_bootstrap`
- `structure_score` e `structure_score_bootstrap`

## 8.2 Tabelas de significância

Em `_build_significance_tables`:

- calcula deltas vs bootstrap
- estima z-score e p-value (aproximação normal)
- gera share de significância (`p < 0.05`) por janela

Arquivo:

- `significance_summary_by_window.csv`

## 8.3 Robustez temporal entre janelas

Em `_build_robustness`:

- consistência de direção de `dp1_5` e `ddeff_5`
- maioria entre janelas
- métricas como `joint_majority_60d`

---

## 9. Diagnóstico por ativo e setor

## 9.1 Por ativo

Função: `_build_asset_sector_diagnostics`

Principais componentes:

- `vol60_latest`, `corr120_abs_latest`, `switches_90d`
- risco por ativo:

`risk_score = 0.45*rank_vol + 0.35*rank_corr_abs + 0.20*rank_sw90`

- confiança por ativo:

`confidence_score = 1 - (0.60*rank_sw90 + 0.40*rank(|delta_corr120_5d|))`

- regime do ativo (`estavel`, `transicao`, `instavel`) via `_state_from_risk`

## 9.2 Por setor

Agregação por setor:

- média de risco
- média de confiança
- `% instável`, `% transição`
- médias de switches

Nível de alerta setorial:

- vermelho se `% instável >= 20%` ou `risk_mean >= 0.75`
- amarelo se `% instável >= 10%` ou `risk_mean >= 0.60`
- verde caso contrário

---

## 10. Alertas operacionais

## 10.1 Alertas de evento

Função `_build_operational_alerts` gera eventos como:

- mudança de regime,
- quebra de robustez,
- outras condições estruturais.

## 10.2 Níveis (verde/amarelo/vermelho)

Função `_build_alert_levels`:

- usa risco e confiança com quantis da política,
- aplica persistência temporal (`persist_window`, `persist_count`).

---

## 11. QA e fronteira de validade

## 11.1 QA obrigatório

Função `_qa` valida:

- consistência do núcleo de ativos,
- cobertura e suficiência por janela,
- mínimo de ativos usados por data,
- razão `q = T / N_used` acima do mínimo,
- faixa válida de `p1` e `deff`,
- completude dos últimos 20 dias.

## 11.2 Fronteira matemática prática

Não emitir leitura confiável quando:

- `N_used` cai abaixo do mínimo,
- `q < q_min`,
- fração de universo insuficiente excede teto,
- janela oficial não está presente.

---

## 12. Gate de publicação

Função `_build_deployment_gate`.

Bloqueia publicação quando qualquer regra crítica falha.

Principais bloqueios:

- `qa_failed`
- `robustness_joint_majority_below_threshold`
- `baseline_universe_mismatch`
- `baseline_delta_p1_exceeds_threshold`
- `baseline_delta_deff_exceeds_threshold`
- `too_many_cluster_alerts`
- `policy_lock_mismatch` (quando exigido)

Resultado:

- `blocked: true/false`
- `reasons: []`
- `checks: {}`
- `thresholds: {}`

---

## 13. Baseline e comparabilidade histórica

Função `_freeze_baseline` mantém:

- hash do universo,
- último estado oficial,
- deltas vs run anterior.

Objetivo:

- impedir drift silencioso,
- garantir comparabilidade de releases,
- detectar mudança de universo antes de publicar.

---

## 14. Backtest de exposição

Função `_backtest`.

- retorno de referência: média simples do universo (`mkt`)
- exposição vem do regime
- execução com lag (causal)
- custo de transação (bps)
- cap de turnover diário

Saídas:

- curva estratégia e benchmark
- retorno anual, vol, sharpe, drawdown
- turnover médio e custo total

Leitura correta:

- Eigen Engine e primariamente diagnostico de risco estrutural,
- não deve ser vendido como promessa de alpha constante.

---

## 15. Event study e validação de antecipação

Script base:

- `scripts/bench/event_study_validate.py`

Regras principais:

- separa treino e teste,
- calibra thresholds no treino,
- define eventos automaticamente,
- mede recall, precisão, falso alarme/ano, antecedência,
- compara com baseline aleatório de mesma frequência de alerta.

Critério de leitura:

- se ganha dos baselines simples com falso alarme controlado, há utilidade operacional;
- se não ganha do aleatório com significância, não pode haver promessa forte de antecipação.

---

## 16. Contrato de API do Eigen Engine no app

Rota:

- `GET /api/lab/corr/latest`

Campos principais:

- `run`
- `metrics.latest_state`
- `regime_history`
- `asset_diagnostics`
- `sector_diagnostics`
- `summary`
- `qa_checks`
- `limits`

Regras:

- API devolve contrato vazio estruturado quando não há run válido,
- front não deve inferir sinal fora desse contrato.

---

## 17. Interface: interpretação correta

No app, o usuário deve ler nesta ordem:

1. gate (`ok` ou `bloqueado`)
2. regime agregado
3. confiança média
4. distribuição de estados por ativo/setor
5. séries históricas (bolinhas/regime line)

Evitar:

- usar somente uma métrica isolada,
- interpretar risco alto como ordem de compra/venda.

---

## 18. Linguagem regulatória segura

Obrigatório manter texto sem imperativo financeiro.

Frases permitidas:

- "leitura estrutural sugerida"
- "diagnóstico de risco"
- "suporte quantitativo"

Frases proibidas no produto:

- "compre"
- "venda"
- "garantia de retorno"
- "previsão certeira de crash"

---

## 19. Como operar no dia a dia

## 19.1 Rotina mínima

1. executar pipeline diário
2. checar `qa_checks` e `deployment_gate.json`
3. validar `latest_release.json`
4. publicar somente se gate liberado

## 19.2 Se gate bloquear

- não forçar publicação,
- inspecionar motivo específico,
- corrigir dado/política,
- rerodar e revalidar.

---

## 20. Troubleshooting rápido

## 20.1 "Sem dados na dashboard"

Checar:

- `public/data/lab_corr_macro/latest`
- integridade do `latest_release.json`
- retorno da rota `/api/lab/corr/latest`

## 20.2 "Regime oscilando demais"

Checar:

- `hysteresis_days`
- thresholds de transição
- ruído de cobertura por janela

## 20.3 "Setor quebrado"

Checar:

- mapeamento de setor no universo
- ativos sem setor (`Sem setor`)
- normalização de nomes

---

## 21. Auditoria pós-morte (post-mortem)

Quando houver alerta controverso, capturar:

- `run_id`
- `policy_hash`
- `qa_checks`
- `deployment_gate`
- `regime_series_T120.csv`
- `asset_regime_diagnostics.csv`
- `sector_regime_diagnostics.csv`

Perguntas padrão:

1. O gate estava liberado?
2. Houve quebra de baseline?
3. A leitura foi consistente entre janelas?
4. O alerta foi episódio isolado ou persistente?

---

## 22. Controle de mudanças

Toda mudança de regra deve:

- atualizar política versionada,
- registrar impacto em robustez e event study,
- comparar contra baseline anterior,
- manter audit trail.

---

## 23. Roadmap técnico do núcleo (sem perder avanço)

Prioridades recomendadas:

1. reforçar teste de integridade temporal contínuo
2. ampliar painel de significância por janela
3. consolidar módulo de calibração com governança mais rígida
4. manter simplificação de produto (painel único)

---

## 24. Índice de referência rápida

- política oficial: `config/lab_corr_policy.json`
- núcleo de execução: `scripts/lab/run_corr_macro_offline.py`
- API de consumo: `website-ui/app/api/lab/corr/latest/route.ts`
- operação diária: `docs/operacao/ROTINA_DIARIA_MASTER.md`
- fluxo ops/publicação: `docs/OPS_EXECUTION_FLOW.md`
- guia de venda: `docs/OFERTA_COMERCIAL_MOTOR.md`

---

## 25. Resumo executivo honesto

O Eigen Engine, no estado atual, e forte em:

- leitura estrutural do mercado,
- rastreabilidade e governança de publicação,
- separação de níveis de risco por ativo e setor.

Limite estrutural inevitável:

- não há previsão perfeita de evento exógeno.

Valor real:

- reduzir cegueira de risco,
- padronizar leitura de contexto,
- criar rotina auditável para decisão humana.

---

## Apêndice A — Fórmulas de decisão (compacto)

- `transition_score = 0.45*z(|dp1_5|) + 0.45*z(|ddeff_5|) + 0.10*z(overlap_instability)`
- `stress` se `p1>=q80` e `deff<=q20`
- `dispersion` se `p1<=q20` e `deff>=q80`
- `transition` se `|dp1_5|>=q80` OU `|ddeff_5|>=q80` OU `transition_score>=q80`
- histerese: persistência mínima para troca de regime

- `risk_asset = 0.45*rank(vol60) + 0.35*rank(|corr120|) + 0.20*rank(switches90)`
- `conf_asset = 1 - (0.60*rank(switches90) + 0.40*rank(|delta_corr120_5d|))`

---

## Apêndice B — Checklist de release

Antes de release:

- [ ] QA `ok=true`
- [ ] `deployment_gate.blocked=false`
- [ ] sem mismatch de universo
- [ ] sem violação de delta baseline
- [ ] build do site sem erro
- [ ] API `/api/lab/corr/latest` respondendo contrato completo

Depois do release:

- [ ] validar dashboard em produção
- [ ] validar tabela de ativos e setores
- [ ] registrar run no log operacional

---

## Apêndice C — Glossário curto

- `p1`: fração da variância no primeiro modo espectral
- `deff`: dimensão efetiva do sistema
- `walk_forward`: calibração causal sem olhar futuro
- `histerese`: regra de permanência mínima para troca de estado
- `gate`: bloqueio/liberação de publicação
