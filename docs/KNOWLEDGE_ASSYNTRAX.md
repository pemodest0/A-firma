# KNOWLEDGE_EIGEN_ENGINE

Base semantica minima para agentes de IA que interpretam o Eigen Engine da Assyntrax.

## Glossario

- `phi`:
  Parametro de ordem estrutural (`lambda1 / soma(lambda_i)`). Sobe quando o modo coletivo domina mais variancia.
- `deff`:
  Dimensao efetiva espectral (`exp(entropia espectral)`). Cai quando a estrutura fica concentrada.
- `entropia espectral`:
  Entropia de probabilidades dos autovalores normalizados. Mede dispersao estrutural.
- `CSD` (critical slowing down):
  Familia de sinais de perda de resiliencia (ex.: variancia e autocorrelacao lag-1 crescentes).
- `curvatura (Forman-Ricci)`:
  Medida topologica local de fragilidade da rede de correlacao. Curvatura media mais negativa sugere maior propensao a propagacao de choque.
- `baseline`:
  Referencia neutra (ruido/benchmark) usada para comparar se ha estrutura real.
- `causalidade`:
  Regra de nao usar dados futuros em calibracao, threshold ou validacao.

## Como interpretar (resumo operacional)

- `phi` subindo de forma sustentada:
  Aumenta sincronizacao estrutural do sistema.
- `deff` caindo:
  Menos graus efetivos de liberdade; sistema mais concentrado.
- `ac1_phi` subindo:
  Pode indicar desaceleracao critica e menor taxa de recuperacao.
- `forman_mean` caindo:
  Pode indicar fragilidade topologica crescente.
- `structural_score` subindo:
  Maior nivel de fragilidade estrutural agregada (nao e previsao de preco).

## Impactos estruturais (ativo/setor/global)

- `impact_global`:
  Contribuicao estrutural do ativo ao modo global (`v1_i^2` no autovetor dominante global). Soma diaria dos ativos = 1.
- `impact_sector`:
  Contribuicao estrutural do ativo ao modo dominante do setor (`v1_i^2` no autovetor setorial). Soma diaria por setor = 1.
- `sector_loading`:
  Impacto do setor no global (soma de `impact_global` dos ativos do setor).
- `overlap_sector_global`:
  Alinhamento do modo setorial com o modo global (alto = maior sincronizacao setorial com sistema).
- `overlap_ab` (setor A vs setor B):
  Similaridade estrutural entre modos setoriais; alta subida simultanea sugere acoplamento entre setores.

Leitura causal minima:

- Ativo `impact_global` subindo + `deff` global caindo: concentracao sistêmica crescente.
- `sector_loading` subindo antes do regime critico: setor potencialmente transmissor de choque.
- `overlap_sector_global` acelerando: setor sendo absorvido pelo modo coletivo.
- `overlap_ab` subindo entre setores: sincronizacao intersetorial.

## O que nao afirmar (anti-hallucination)

- Nao afirmar previsao de preco-alvo.
- Nao afirmar data exata de crash.
- Nao afirmar causalidade economica sem teste dedicado.
- Nao afirmar robustez fora do dominio validado.
- Nao afirmar que backtest garante resultado futuro.

## Regras de citacao de evidencias

- Referenciar sempre artefatos reais em `results/...`.
- Citar `run_id` e arquivo origem ao resumir resultado.
- Se dado estiver ausente, responder explicitamente que nao ha evidencia no run atual.

## Ground truth oficial (estrutural v1)

- Script de avaliacao:
  `scripts/structural/run_ground_truth_tests.py`
- Saidas:
  - `results/structural_ground_truth_<timestamp>/ground_truth_summary.json`
  - `results/structural_ground_truth_<timestamp>/ground_truth_daily.csv`
  - `results/ops/ai_knowledge/latest_ground_truth.json`

Definicao de evento-verdade no v1 (finance):

- evento = drawdown futuro de benchmark abaixo de limiar em horizonte fixo
- evento alternativo = entrada futura em regime critico (`stress` ou `transition`) no horizonte
- base padrao: horizontes `5,10,20` dias e limiar de drawdown `5%`

Metricas padrao para IA citar:

- precision
- recall
- f1
- accuracy
- event_rate
- alert_rate

## Artefatos oficiais de impacto/IA

- Script:
  `scripts/structural/run_structural_impact_learning.py`
- Saidas:
  - `asset_impact_daily.csv`
  - `sector_impact_daily.csv`
  - `sector_pair_overlap_daily.csv`
  - `impact_training_dataset.csv`
  - `impact_model_eval.csv`
  - `impact_summary.json`
  - `results/ops/ai_knowledge/latest_structural_impact.json`

## Briefing operacional unificado (IA)

- Script:
  `scripts/ops/build_ai_operational_brief.py`
- Saidas:
  - `results/ops/ai_knowledge/operational_brief_<timestamp>.json`
  - `results/ops/ai_knowledge/latest_operational_brief.json`
- Campos minimos para consumir:
  - `data_last_date`
  - `freshness.status` e `freshness.days_lag`
  - `operational_signal.risk_level_next_month`
  - `operational_signal.operational_state`
  - `operational_signal.action_hint`
  - `model_evidence.ground_truth_best`
  - `model_evidence.horizon_winners_global`
  - `state_snapshot.global_state`
  - `state_snapshot.top_sectors_global_mode`
  - `state_snapshot.top_assets_global_mode`

## Protocolo de insights para IA

- Insight valido precisa conter:
  - `mensagem`
  - `evidence` com metricas numericas
  - referencia de arquivo origem
- Priorizar insights que combinem:
  - estado global (score/phi/deff)
  - ranking setorial (loading/overlap)
  - robustez historica (f1/lift)
- Evitar insight sem contraponto de incerteza quando:
  - event_rate muito baixo
  - alert_rate muito alto
  - base desatualizada (`freshness=attention|stale`)
