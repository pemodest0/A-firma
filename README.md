# Plataforma Integrada de Simulacoes

Base unica para pipelines financeiros quanticos, operadores GA/Lie e visualizacoes interativas. Tudo foi organizado seguindo os blocos pedidos para que seja facil descobrir onde ficam scripts, dados, modelos e paines.

## Estrutura
- `scripts/` – CLIs e utilitarios rapidos. `scripts/estudos/` guarda experimentos mais pesados (stress tests, comparacao de metricas, benchmarks).
- `visualizacao/` – dashboards Streamlit/Dash sob `visualizacao/dashboards/`, notas em `visualizacao/docs/` e apps dedicados (ex.: `visualizacao/streamlit/simulador_qubits_algebras.py`).
- `modelos/` – todo o codigo de modelos. `modelos/core/src/` contem os modulos `data_pipeline`, `walks`, etc.; `modelos/gaq/` concentra Geometric Algebra; `modelos/tests/` cobre ingestion/quality/hybrid forecast.
- `dados/` – `dados/brutos/` guarda CSVs, NPYs e scripts auxiliares (como `financial_loader.py`), `dados/configs/` concentra os JSONs das pipelines e `dados/benchmarks/` armazena artefatos exportados pelos estudos.
- `agent/` – espaco reservado para integrar workflows GPT/LLM (por enquanto apenas README).
- `requirements.txt` – dependencias principais da plataforma.
- `.streamlit/config.toml` – tema escuro padrao para todos os paines Streamlit.

A raiz tambem possui `sitecustomize.py`, que injeta automaticamente `modelos/` e `modelos/core/` no `sys.path`. Assim, os imports `import src...` e `import gaq...` continuam funcionando mesmo apos a reorganizacao.

## Fluxo rapido
- **Pipeline financeiro completo**  
  `python scripts/run_pipeline.py --config dados/configs/data_pipeline_finance.json`

- **Previsao diaria (arquivo CSV local ou Yahoo Finance)**  
  `python scripts/run_daily_forecast.py --config dados/configs/live_forecast_template.json`

- **Descoberta de grafos**  
  `python scripts/run_graph_discovery_finance.py`

- **Dashboards**  
  `streamlit run visualizacao/dashboards/app/asset_forecast_dashboard.py` ou `python -m visualizacao.dashboards.app.quantum_explorer`.

- **Simulacoes GA/Lie**  
  `streamlit run visualizacao/streamlit/simulador_qubits_algebras.py`

- **Testes**  
  `pytest modelos/tests modelos/gaq/tests`

- **Estudos/benchmarks**  
  `python scripts/estudos/stress_pipeline.py --domains all`  
  `python scripts/estudos/project_walk_embeddings.py --n 4 --step 40`

## Dados e higiene
- Todos os CSV/NPY vivem em `dados/brutos/`. Subpastas importantes:
  - `dados/brutos/yf/` – dumps do Yahoo Finance baixados via `scripts/download_from_yf.py`.
  - `dados/brutos/metrics/` – saidas de `scripts/run_metric_*.py`.
  - `dados/brutos/financial_loader.py` – helper unico para baixar/normalizar precos.
- Resultados de benchmarks dos estudos estao em `dados/benchmarks/`.
- `results/` segue reservado para saídas volumosas (forecasts, figuras, logs). Armazene ali e mova para `results/archive/` quando necessario.
- Limpeza rapida:
  ```powershell
  Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
  Get-ChildItem -Recurse -Filter '*.pyc' -File | Remove-Item -Force
  Get-ChildItem -Recurse -Filter '.DS_Store' -File | Remove-Item -Force
  ```

## Modelos e GAQ
- O nucleo python continua identico, apenas realocado para `modelos/core/src/`. Os scripts que dependem de `src.*` ja usam `sitecustomize`/`PYTHONPATH` ajustado.
- `modelos/gaq/` manteve a mesma arvore (analysis, core, walks, tests, examples). `pyproject.toml` agora procura pacotes dentro de `modelos/`, entao `python -m pytest modelos/gaq/tests` funciona direto.
- Para instalar dependencias de desenvolvimento: `pip install -r requirements.txt` (adiciona numpy, pandas, scipy, streamlit, plotly, networkx, sklearn, requests, yfinance, clifford).

## Visualizacao
- Dashboards Streamlit/Dash residem em `visualizacao/dashboards/app/`.
- As figuras estaticas (por ex. `geom_distribution_uniform.png`) ficam em `visualizacao/docs/assets/` junto com os markdowns.
- `.streamlit/config.toml` unifica cores, de modo que qualquer `streamlit run ...` herda o mesmo tema.

## Scripts de estudos
- `scripts/estudos/` concentra:
  - `stress_pipeline.py` – orquestra rodadas completas em dominios finance/health/logistics/physics.
  - `metrics_comparison.py`, `make_report.py`, `run_benchmarks.py` – constroem tabelas e PDFs.
  - `evaluate_walk_metrics.py` e `project_walk_embeddings.py` agora escrevem em `dados/benchmarks/metrics_output` e `dados/benchmarks/embed_output`.

## Agente
`agent/README.md` descreve como plugar futuros copilots (workflow, automacoes, etc.). Quando existir codigo efetivo, coloque notebooks/lambdas aqui para manter o resto limpo.

## Observacoes finais
- `requirements.txt` cobre apenas dependencias runtime; ferramentas de lint/teste continuam declaradas no `pyproject`.
- `pyproject.toml` aponta o discovery do setuptools para `modelos/`, e o `pytest` ja inicializa com `modelos/core` no caminho.
- `A-firma.zip` e demais duplicatas foram removidas para evitar lixos grandes no Git.
