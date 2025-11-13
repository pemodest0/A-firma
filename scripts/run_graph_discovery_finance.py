#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_pipeline import FinanceImportConfig, load_finance_dataset
from graph_discovery import graph_registry, select_best_graph, GraphPenalties, GraphCandidate, summarize_walk
from meta import extract_walk_features
from walks import run_classical_walk

CONFIG_PATH = Path("dados/configs/data_pipeline_finance.json")
OUTPUT_DIR = Path("results/graph_discovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_finance_dataset(config: Dict) -> pd.DataFrame:
    finance_cfg = config["finance"]
    sources = [Path(src) for src in finance_cfg["sources"]]
    import_config = FinanceImportConfig(
        sources=sources,
        tickers=finance_cfg.get("tickers"),
        min_history=int(finance_cfg.get("min_history", 250)),
        compute_returns=bool(finance_cfg.get("compute_returns", True)),
    )
    dataset = load_finance_dataset(import_config)
    if dataset.empty:
        raise ValueError("Finance dataset resultou vazio após filtragem.")
    max_rows = finance_cfg.get("max_rows")
    if max_rows:
        dataset = dataset.tail(int(max_rows)).reset_index(drop=True)
    return dataset


def _graph_label_smoothness(candidate: GraphCandidate, label: np.ndarray) -> float:
    adjacency = candidate.graph.adjacency
    if "vertex_assignments" in candidate.metadata:
        assignments = candidate.metadata["vertex_assignments"]
        num_vertices = adjacency.shape[0]
        vertex_sum = np.zeros(num_vertices, dtype=float)
        counts = np.zeros(num_vertices, dtype=float)
        for idx, vertex in enumerate(assignments):
            vertex_sum[vertex] += label[idx]
            counts[vertex] += 1.0
        means = np.divide(
            vertex_sum,
            np.where(counts == 0, 1.0, counts),
        )
        contribution = 0.0
        edge_count = 0.0
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if adjacency[i, j]:
                    diff = means[i] - means[j]
                    contribution += diff * diff
                    edge_count += 1.0
        return float(contribution / edge_count) if edge_count else float("inf")
    else:
        if adjacency.shape[0] != label.shape[0]:
            raise ValueError("Dimensão de label não combina com grafo.")
        contribution = 0.0
        edge_count = 0.0
        for i in range(adjacency.shape[0]):
            neighbors = np.where(adjacency[i] > 0)[0]
            for j in neighbors:
                if j <= i:
                    continue
                diff = label[i] - label[j]
                contribution += diff * diff
                edge_count += 1.0
        return float(contribution / edge_count) if edge_count else float("inf")


def make_candidates(config: Dict, dataset: pd.DataFrame) -> List[GraphCandidate]:
    candidates = []
    for entry in config.get("graph_candidates", []):
        family = entry["family"]
        params = entry.get("params", {})
        if family not in graph_registry:
            raise KeyError(f"Família de grafo desconhecida: {family}")
        builder = graph_registry[family]
        candidate = builder(dataset, **params)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("Nenhum candidato de grafo definido no config.")
    return candidates


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset = build_finance_dataset(config)

    label_column = config.get("selection", {}).get("label_column", "return_t1")
    if label_column not in dataset.columns:
        raise KeyError(f"Coluna de label '{label_column}' ausente no dataset.")

    feature_union = sorted(
        {
            feat
            for entry in config.get("graph_candidates", [])
            for feat in entry.get("params", {}).get("feature_columns", [])
        }
    )
    columns_to_check = feature_union + [label_column]
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    if columns_to_check:
        dataset = dataset.dropna(subset=columns_to_check).reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Dataset ficou vazio após remoção de NaNs.")
    label = dataset[label_column].to_numpy(dtype=float)

    candidates = make_candidates(config, dataset)

    penalties_cfg = config.get("selection", {}).get("penalties", {})
    penalties = GraphPenalties(
        density=float(penalties_cfg.get("density", 0.0)),
        short_cycles=float(penalties_cfg.get("short_cycles", 0.0)),
        diameter=float(penalties_cfg.get("diameter", 0.0)),
    )

    def score_fn(candidate: GraphCandidate) -> float:
        smoothness = _graph_label_smoothness(candidate, label)
        return smoothness

    best_candidate, best_score = select_best_graph(candidates, score_fn, penalties)

        target_cfg_path = Path("dados/configs/targets_finance.json")
    threshold_values = (0.5, 0.3, 0.1)
    aggregate_top_k = 5
    custom_target_nodes = None
    custom_target_info = None
    if target_cfg_path.exists():
        target_cfg = json.loads(target_cfg_path.read_text())
        threshold_values = tuple(target_cfg.get("fallback_thresholds", threshold_values))
        aggregate_top_k = int(target_cfg.get("aggregation_top_k", aggregate_top_k))
        thresholds = target_cfg.get("cluster_thresholds", {})
        mask = np.ones(len(dataset), dtype=bool)
        if "mom_high" in thresholds:
            mask &= dataset["mom"] > float(thresholds["mom_high"])
        if "vol_ratio_high" in thresholds:
            mask &= dataset["vol_ratio"] > float(thresholds["vol_ratio_high"])
        if "drawdown_low" in thresholds:
            mask &= dataset["drawdown"] < float(thresholds["drawdown_low"])
        vertex_assignments = best_candidate.metadata.get("vertex_assignments")
        if vertex_assignments is not None:
            custom_target_nodes = tuple(sorted({int(vertex_assignments[idx]) for idx, flag in enumerate(mask) if flag}))
            custom_target_info = {
                "target_condition_count": int(mask.sum()),
                "target_condition_fraction": float(mask.mean()),
            }

    walk_result = run_classical_walk(best_candidate.graph, steps=10, start_node=0)
    walk_metrics = extract_walk_features(walk_result)
    walk_summary = summarize_walk(
        walk_result,
        thresholds=threshold_values,
        aggregate_top_k=aggregate_top_k,
        target_nodes=custom_target_nodes,
    )

    summary = {
        "best_family": best_candidate.family,
        "best_params": best_candidate.params,
        "best_score": best_score,
        "penalties": asdict(penalties),
        "walk_metrics": walk_metrics,
        "walk_summary": walk_summary,
        "custom_target_info": custom_target_info,
        "dataset_rows": int(dataset.shape[0]),
        "label_column": label_column,
    }

    output_path = OUTPUT_DIR / "finance_graph_selection.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Grafo selecionado:", summary["best_family"])
    print("Score penalizado:", summary["best_score"])
    print("Resumo salvo em:", output_path)


if __name__ == "__main__":
    main()
