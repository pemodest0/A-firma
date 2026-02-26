# Eigen Engine - Mapa de Arquivos e Funcoes

Status: ativo  
Atualizado em: 2026-02-25

Objetivo: manter um inventario tecnico do motor com responsabilidade por arquivo e funcao.

## 1) Nucleo do motor (pipeline estrutural)

### `scripts/lab/run_corr_macro_offline.py`
Responsabilidade: executar o motor estrutural offline fim-a-fim (janelas, regime, alertas, QA, gate e artefatos).

#### Funcoes utilitarias
- `_run_id`: gera run id UTC.
- `_find_latest_finance_run`: resolve run financeiro mais recente.
- `_ensure_cols`: valida colunas obrigatorias de DataFrame.
- `_safe_sign`: sinal robusto para arrays com tolerancia numerica.
- `_parse_windows`: parse da lista de janelas (`60,120,252`).

#### Funcoes matematicas de base
- `_cluster_metrics`: extrai metrica de cluster na matriz de correlacao.
- `_turnover_pair_frac`: turnover entre particoes/clusters.
- `_spectral_metrics`: calcula `p1`, `deff`, `top5`, `entropy`.
- `_spectral_metrics_with_v1`: idem acima com autovetor dominante (`v1`).
- `_zscore_series`: z-score simples.
- `_zscore_expanding`: z-score causal em janela expanding.
- `_expanding_quantile`: quantil causal expanding.
- `_block_bootstrap_col`: bootstrap em bloco por serie.
- `_block_bootstrap_matrix`: bootstrap em bloco para matriz de retornos.

#### Pipeline de janela e consolidacao
- `_process_window`: calcula serie estrutural por janela (core do calculo diario).
- `_summary_block`: gera resumo textual/estrutural da janela.
- `_majority_same_direction`: consistencia de direcao entre janelas.
- `_build_robustness`: robustez cruzando janelas (majority/consenso).
- `_apply_hysteresis`: estabiliza trocas de estado com persistencia minima.
- `_classify_regime`: classificador oficial de regime (walk-forward + histerese).

#### Performance, alerta e nivel
- `_perf`: metricas de retorno/risco para backtest.
- `_backtest`: backtest por regime e exposicao.
- `_cluster_alerts`: alertas baseados em clusterizacao estrutural.
- `_era_name`: normaliza nome da era/periodo.
- `_build_operational_alerts`: cria alertas operacionais diaros.
- `_apply_level_persistence`: persistencia de nivel de alerta (2-em-3).
- `_build_alert_levels`: gera niveis de alerta por data.

#### Significancia e diagnosticos
- `_normal_two_sided_p`: p-valor bilateral aproximado (normal).
- `_build_significance_tables`: tabela de significancia por janela.
- `_switch_count`: conta trocas de estado.
- `_state_from_risk`: estado sintetico a partir de risco/confianca/switch.
- `_build_asset_sector_diagnostics`: diagnostico por ativo e setor.
- `_build_era_evaluation`: avaliacao por eras (stress/normal/transicao).

#### Playbook, UI e QA
- `_action_map`: mapeia regime para acao operacional.
- `_reliability_tier`: classifica confianca em tier.
- `_build_action_playbook`: gera trilha operacional diaria.
- `_build_ui_view_model`: payload consolidado para UI/API.
- `_qa`: checagens de qualidade e consistencia do run.
- `_freeze_baseline`: congela baseline de comparacao.
- `_build_deployment_gate`: avalia gate final de publicacao.
- `_write_compact_report`: escreve relatorio compacto de run.

#### Politica, lock e release
- `_dict_hash`: hash deterministico de payload.
- `_build_policy_lock`: lock de politica aplicada no run.
- `_update_release_pointer`: move ponteiro de release quando gate aprova.
- `_load_policy`: carrega politica oficial JSON.
- `_slug_token`: normaliza token para nome de arquivo/chave.
- `_resolve_baseline_dir`: resolve baseline ativo para comparacao.
- `_apply_policy_to_args`: aplica politica sobre argumentos da CLI.

#### Calibracao e casos
- `_default_exposure_candidates`: grid padrao de exposicao para busca.
- `_score_backtest`: score objetivo do backtest para ranking.
- `_exposure_grid_search`: busca de exposicao candidata.
- `_write_daily_brief`: resumo diario para operacao.
- `_write_commercial_narrative`: narrativa comercial derivada dos resultados.
- `_parse_csv_list`: parse de listas CSV em string.
- `_future_stats`: calcula estatisticas futuras apos ancora.
- `_case_score`: score interno por caso/regime.
- `_choose_case_row`: escolhe linha representativa do caso.
- `_honest_verdict`: veredito textual sem overclaim.
- `_build_case_studies`: gera estudos de caso estruturados.
- `_write_case_studies_demo`: exporta demo resumida de casos.

#### Bloco hierarquico (global + setor)
- `_build_structural_parallel_diagnostics`: diagnosticos estruturais paralelos.
- `_write_vectors`: salva vetores dominantes (`v1`) por data.
- `_load_hierarchical_metadata`: carrega/enriquece metadata setorial.
- `_compute_cross_daily`: metrica cross setor-global diaria.
- `_write_hierarchical_state`: estado hierarquico pronto para produto.
- `_run_hierarchical_diagnostics`: executa pipeline hierarquico completo.

#### Entrada principal
- `main`: parser CLI e orquestracao completa do run.

## 2) Biblioteca estrutural reutilizavel

### `engine/structural/rmt.py`
- `mp_bounds`: limites de Marcenko-Pastur (`lambda_min`, `lambda_max`).
- `significant_eigs`: autovalores acima de `lambda_max`.
- `rmt_report`: resumo de sinal vs ruido espectral.

### `engine/structural/spectral.py`
- `normalize_eigs`: normaliza autovalores com clipping numerico.
- `spectral_entropy`: entropia espectral.
- `effective_dimension`: dimensao efetiva (`deff`).
- `order_param_phi`: parametro de ordem (`phi`).
- `spectral_pack`: pacote consolidado (`phi`, `H`, `deff`, `lambda1`, `topk`).

### `engine/structural/csd.py`
- `_to_series`: coercao para serie.
- `rolling_variance`: variancia rolling causal.
- `_ac1_window`: autocorrelacao lag-1 por janela.
- `rolling_ac1`: serie de `ac1` rolling causal.
- `_zscore_against_train`: z-score usando baseline de treino.
- `ews_pack`: pacote CSD (var, ac1, zscores).

### `engine/structural/graph.py`
- `corr_to_graph`: converte correlacao em grafo (`topk` ou limiar).

### `engine/structural/forman_ricci.py`
- `forman_edge_curvature`: curvatura de Forman por aresta.
- `forman_summary`: resumo estatistico da curvatura.

### `engine/structural/score.py`
- `fit_normalizer`: ajusta normalizador no treino.
- `transform`: aplica normalizacao/zscore em serie.
- `structural_score`: score estrutural combinado (`phi`, `deff`, `ac1`, curvatura).

### `engine/structural/impact.py`
- `_as_vector_df`: padroniza formato de vetor estrutural.
- `_normalize_square_by_group`: normaliza contribuicoes quadradas por grupo.
- `compute_asset_global_impact`: impacto ativo->global (`v1^2`).
- `compute_asset_sector_impact`: impacto ativo->setor.
- `merge_asset_sector_global_impacts`: merge de impactos para produto.
- `compute_sector_pair_overlap`: overlap estrutural setor<->setor.

### `engine/structural/ground_truth.py`
- `forward_max_drawdown_from_equity`: drawdown maximo futuro por horizonte.
- `build_event_label`: rotulo binario de evento futuro.
- `build_regime_future_event_label`: rotulo futuro condicionado a regime.
- `classification_report_binary`: precision/recall/f1/lift.
- `threshold_from_train`: limiar calibrado apenas no treino.

### `engine/structural/run_manifest.py`
- `utc_timestamp`: timestamp UTC deterministico.
- `git_hash`: hash curto do commit corrente.
- `_normalize_gates`: normaliza status dos gates.
- `write_run_manifest`: escreve `RUN_MANIFEST.json` auditavel.

## 3) Selecao de universo e metadata

### `engine/core/universe.py`
- `_prepare_metadata`: valida e padroniza metadata de ativos.
- `_coverage_liquidity_table`: cobertura/liquidez por ativo.
- `_rank_assets`: ranking deterministico para selecao.
- `select_global_universe`: universo global (N alvo + cobertura minima).
- `select_sector_universe`: universo setorial (N alvo + cobertura minima).

### `engine/ops/metadata.py`
- `load_asset_metadata`: leitura/validacao de `asset_metadata.csv`.

## 4) Orquestracao operacional (pipeline diario)

### `scripts/ops/run_daily_master.py`
Responsabilidade: orchestrator oficial do dia (chama motor, valida, gera snapshots e consolida estado operacional).

### `scripts/ops/publish_latest_if_gate_ok.py`
Responsabilidade: publica apenas quando `deployment_gate` aprova.

### `scripts/ops/run_daily_validation.py`
Responsabilidade: executa o bloco oficial de validacao diaria e consolida `results/ops/daily/<run_id>/summary.json`.

## 5) Regras de governanca para atualizacao deste mapa

Em todo commit que altere funcao do motor:
1. Atualizar este arquivo com nova funcao/arquivo alterado.
2. Atualizar `README.md` quando houver mudanca de fluxo canonico.
3. Garantir que nome oficial permaneceu: `Assyntrax` (empresa) e `Eigen Engine` (motor).
