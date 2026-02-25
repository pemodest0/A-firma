# Eigen Engine Freeze (Snapshot)

Data: 2026-02-07

Objetivo: congelar o estado funcional do Eigen Engine para referencia, auditoria e retomada.

## Componentes principais (core)
- Embedding + seleção de parametros:
  - `graph_engine/embedding.py`
  - AMI (tau) + Cao/FNN (m) + Takens embedding
- Diagnósticos dinâmicos:
  - `graph_engine/diagnostics.py`
  - AMI + Cao + RQA (DET/LAM/TT) + métricas auxiliares
- Microestados e smoothing:
  - `graph_engine/microstates.py`
  - KMeans/HDBSCAN + HMM smoothing (labels e noise)
- Labels e classificação de regime:
  - `graph_engine/labels.py`
  - Suavização de labels + mapeamento de regimes
- Motor principal (pipeline):
  - `graph_engine/core.py`
  - `graph_engine/metastable.py`
  - `graph_engine/plots.py`
  - `graph_engine/export.py`
  - `graph_engine/report.py`

## Scripts de benchmark e execução
- Universo (batch):
  - `scripts/bench/run_graph_regime_universe.py`
- Hypertest:
  - `scripts/bench/run_graph_regime_hypertest.py`
- Backtest:
  - `scripts/bench/run_graph_regime_backtest.py`
- Forecast:
  - `scripts/bench/run_graph_regime_forecast_train.py`
- Comparação com regimes oficiais:
  - `scripts/bench/compare_official_regimes.py`

## Ajustes recentes relevantes
- Smoothing HMM em microestados e labels:
  - `graph_engine/microstates.py` e `graph_engine/labels.py`
  - CLIs: `--micro-smooth hmm` / `--state-smooth hmm`
- Forecast com ensemble “top2” por regime:
  - `scripts/bench/compute_best_models_top2.py`
  - `run_graph_regime_forecast_train.py --use-best-ensemble`

## Outputs essenciais
- Resultados do motor: `results/latest_graph/`
- Forecast: `results/forecast_suite/`
- Benchmarks: `results/hypertest/`, `results/official_regimes/compare/`

## Versão congelada
Este snapshot representa o ponto estavel do Eigen Engine com:
- Embedding automático opcional (AMI/Cao ou ACF/FNN)
- HMM smoothing para reduzir flicker
- Diagnósticos RQA e métricas de estabilidade
- Forecast comparativo por regime

Observacao: este arquivo documenta o estado do Eigen Engine. Nao substitui versionamento em git.
