# KNOWLEDGE_ASSYNTRAX

Base semantica minima para agentes de IA que interpretam o motor Assyntrax.

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
