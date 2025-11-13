# Pipeline de Descoberta de Grafos + Walks

## Visão Geral
1. **Schemas** (`graph_discovery/schemas.py`): funções `load_finance_records`, `load_logistics_records`, `load_health_records` e `load_physics_records` normalizam dados crus em registros estruturados por domínio.
   - Importadores dedicados: `data_pipeline/finance.py`, `data_pipeline/logistics.py`, `data_pipeline/health.py`, `data_pipeline/physics.py` (reexportados em `data_pipeline_import.py`) já retornam dataframes com features chave (`return_t1`, `mom`, `vol_ratio`, `drawdown`, `window_hours`, `lateness`, `value_zscore`, `state_velocity`, etc.) antes do discovery.
   - Métricas em tempo quase-real: scripts dedicados (`scripts/run_metric_finance.py`, `scripts/run_metric_health.py`, `scripts/run_metric_logistics.py`) utilizam `data_pipeline.metrics` para puxar dados externos (Stooq, World Bank, NYC Open Data) e salvar JSONL em `dados/brutos/metrics/<domínio>/`.
2. **Construção de Grafos** (`graph_discovery/builders.py`):
   - `build_hypercube_graph`: discretiza features com Gray code via `HypercubeEncoder`.
   - `build_knn_graph`: liga observações semelhantes em espaço Euclidiano.
   - `build_multilayer_graph`: separa por camada temporal e conecta entidades entre camadas.
   - `build_bipartite_graph`: orders ↔ resources.
   - `build_mst_shortcuts_graph`: MST robusto + atalhos por quantil.
   - `build_line_graph_from_base`: gera line graph para modelar transferências.
3. **Seleção de Topologia** (`graph_discovery/selectors.py`):
   - `penalized_score` aplica penalidades de densidade, ciclos curtos e diâmetro.
   - `select_best_graph` avalia candidatos via score externo (ex.: MAE ou custo logístico).
4. **Walks** (`walks/`):
   - `run_classical_walk` (Markov).
   - `run_discrete_quantum_walk` e `run_continuous_quantum_walk` (DTQW/CTQW).
   - `run_noisy_quantum_walk` (DensityMatrix + canais de ruído).
   - `lie_tools` reexporta métricas SU(2) (Λ).
5. **Meta-Features & Chooser** (`meta/`):
   - `extract_walk_features` computa mixing/hitting/entropias.
   - `WalkChooser` usa centróides para mapear features → modo ({clássico, quântico, aberto}).
6. **Heads de Tarefa** (`heads/`):
   - Finanças: CRPS, eventos extremos e métricas pontuais.
   - Logística: custo esperado, Brier para on-time, robustez com cenários.

## Fluxo Sugerido
```python
from graph_discovery import graph_registry, select_best_graph, GraphPenalties
from walks import run_discrete_quantum_walk
from meta import extract_walk_features, WalkChooser
from heads import evaluate_finance_distribution

graph_candidates = [
    graph_registry["hypercube"](df, feature_columns=["mom", "vol_ratio"]),
    graph_registry["knn"](df, feature_columns=["mom", "vol_ratio"], k=10),
]

def validation_score(candidate):
    # usuário calcula loss do head para o grafo candidato
    return external_validation_loss(candidate)

best_graph, _ = select_best_graph(
    graph_candidates,
    validation_score,
    penalties=GraphPenalties(density=0.5, short_cycles=0.2, diameter=0.1),
)

walk_result = run_discrete_quantum_walk(best_graph.graph, steps=30)
features = extract_walk_features(walk_result, target_node=5)

chooser = WalkChooser()
chooser.fit([features], ["quantum"])
mode = chooser.predict_one(features)
```

## Próximos Passos
- Integrar `validation_score` com scripts existentes de forecast.
- Alimentar o `WalkChooser` com janelas históricas para meta-aprendizado real.
- Conectar `heads/finance.py` e `heads/logistics.py` aos pipelines de avaliação de benchmark.
