#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


FEATURE_NAMES = (
    "mean_confidence",
    "mean_quality",
    "pct_transition",
    "hazard_score",
    "hybrid_ews_score",
    "hybrid_var95_hist",
    "hybrid_ewma_sigma",
    "regime_age_days",
    "changepoint_flag",
    "pseudo_bifurcation_flag",
)
LEVEL_RISK = {"verde": 0.1, "amarelo": 0.55, "vermelho": 0.9}


def _rel_path(value: Path) -> str:
    try:
        return str(value.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(value)


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return float(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _target(mean_conf: float, pct_transition: float, hazard: float) -> float:
    return _clip01(0.40 * _clip01(pct_transition) + 0.35 * _clip01(hazard) + 0.25 * _clip01(1.0 - mean_conf))


def _parse_ts(value: Any, fallback: datetime) -> datetime:
    raw = str(value or "").strip()
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return fallback


def _norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _build_adjacency(Xz: np.ndarray, k: int) -> np.ndarray:
    n = Xz.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.eye(1, dtype=float)
    k_eff = max(1, min(k, n - 1))
    sim = Xz @ Xz.T
    np.fill_diagonal(sim, -1e9)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = np.argpartition(sim[i], -k_eff)[-k_eff:]
        A[i, idx] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    return deg_inv_sqrt @ A @ deg_inv_sqrt


def _extract_panel_samples(panel: dict[str, Any], *, fallback_ts: datetime) -> list[dict[str, Any]]:
    entries = panel.get("entries")
    if not isinstance(entries, list):
        return []

    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        micro = entry.get("micro")
        gates = entry.get("gates")
        macro = entry.get("macro")
        if not isinstance(micro, dict):
            micro = {}
        if not isinstance(gates, dict):
            gates = {}
        if not isinstance(macro, dict):
            macro = {}

        mean_conf = _clip01(_to_float(micro.get("mean_confidence"), 0.5))
        mean_quality = _clip01(_to_float(micro.get("mean_quality"), 0.5))
        pct_transition = _clip01(_to_float(micro.get("pct_transition"), 0.5))
        hazard = _clip01(_to_float(gates.get("hazard_score"), 0.5))
        ews = _clip01(_to_float(gates.get("hybrid_ews_score"), 0.5))
        var95 = _to_float(gates.get("hybrid_var95_hist"), 0.0)
        ewma_sigma = _to_float(gates.get("hybrid_ewma_sigma"), 0.0)
        regime_age = _to_float(gates.get("regime_age_days"), 0.0)
        changepoint = 1.0 if bool(gates.get("changepoint_flag", False)) else 0.0
        pseudo_flag = 1.0 if bool(macro.get("pseudo_bifurcation_flag", False)) else 0.0

        rows.append(
            {
                "features": [
                    mean_conf,
                    mean_quality,
                    pct_transition,
                    hazard,
                    ews,
                    var95,
                    ewma_sigma,
                    regime_age,
                    changepoint,
                    pseudo_flag,
                ],
                "target": _target(mean_conf, pct_transition, hazard),
                "timestamp": _parse_ts(entry.get("timestamp"), fallback_ts),
                "source": "risk_truth_panel",
                "entity": str(entry.get("asset_id", "unknown")),
            }
        )
    return rows


def _load_asset_group_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                asset = str(row.get("asset", "")).strip().upper()
                group = str(row.get("group", "")).strip().lower() or "unknown"
                if asset:
                    out[asset] = group
    except OSError:
        return {}
    return out


def _timestamp_from_path(path: Path) -> datetime:
    text = str(path)
    m_full = re.search(r"(20\d{6}T\d{6}Z)", text)
    if m_full:
        try:
            return datetime.strptime(m_full.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    m_day = re.search(r"(20\d{6})", text)
    if m_day:
        try:
            return datetime.strptime(m_day.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.now(timezone.utc)


def _extract_master_summary_samples(path: Path, *, asset_groups: dict[str, str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    ts = _timestamp_from_path(path)
    rows: list[dict[str, Any]] = []
    by_group: dict[str, list[tuple[float, float, float, float, float]]] = {}

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                status = str(raw.get("status", "ok")).strip().lower()
                if status and status != "ok":
                    continue
                conf = _clip01(_to_float(raw.get("mean_confidence"), 0.5))
                quality = _clip01(_to_float(raw.get("mean_quality"), conf))
                transition = _clip01(_to_float(raw.get("pct_transition"), 0.5))
                hazard = _clip01(0.55 * transition + 0.45 * (1.0 - conf))
                ews = _clip01(0.50 * transition + 0.50 * hazard)
                asset_id = str(raw.get("asset_id", "")).strip()
                if not asset_id:
                    continue

                rows.append(
                    {
                        "features": [conf, quality, transition, hazard, ews, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "target": _target(conf, transition, hazard),
                        "timestamp": ts,
                        "source": f"master_summary_asset:{path.parent.name}",
                        "entity": asset_id,
                    }
                )

                group = asset_groups.get(asset_id.upper(), "unknown")
                by_group.setdefault(group, []).append((conf, quality, transition, hazard, ews))
    except OSError:
        return []

    for group, vals in by_group.items():
        if len(vals) < 3:
            continue
        arr = np.array(vals, dtype=float)
        conf, quality, transition, hazard, ews = np.mean(arr, axis=0).tolist()
        rows.append(
            {
                "features": [conf, quality, transition, hazard, ews, 0.0, 0.0, 0.0, 0.0, 0.0],
                "target": _target(conf, transition, hazard),
                "timestamp": ts,
                "source": f"master_summary_group:{path.parent.name}",
                "entity": f"group:{group}",
            }
        )
    return rows


def _extract_sector_db_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return []

    query = """
    SELECT
      s.sector,
      LOWER(COALESCE(s.alert_level, 'verde')) AS alert_level,
      s.sector_score,
      s.share_unstable,
      s.share_transition,
      s.mean_confidence,
      r.generated_at_utc
    FROM sector_snapshots s
    JOIN runs r ON r.run_id = s.run_id
    ORDER BY r.generated_at_utc ASC, s.sector ASC
    """
    try:
        data = conn.execute(query).fetchall()
    except sqlite3.Error:
        conn.close()
        return []
    conn.close()

    rows: list[dict[str, Any]] = []
    prev_level: dict[str, str] = {}
    regime_age: dict[str, int] = {}
    now = datetime.now(timezone.utc)

    for sector, level, score, share_unstable, share_transition, mean_confidence, ts_raw in data:
        sector_s = str(sector or "unknown")
        lvl = str(level or "verde").lower()

        conf = _clip01(_to_float(mean_confidence, 0.5))
        quality = conf
        transition = _clip01(_to_float(share_transition, 0.5))
        unstable = _clip01(_to_float(share_unstable, transition))
        hazard = _clip01(0.55 * unstable + 0.45 * transition)

        score_f = _to_float(score, hazard)
        ews = _clip01(score_f / 2.0) if score_f > 1.0 else _clip01(score_f)

        prev = prev_level.get(sector_s)
        changed = 1.0 if prev is not None and prev != lvl else 0.0
        age = int(regime_age.get(sector_s, 0))
        if prev is None or prev != lvl:
            age = 0
        else:
            age += 1
        regime_age[sector_s] = age
        prev_level[sector_s] = lvl

        level_risk = _clip01(LEVEL_RISK.get(lvl, 0.5))
        target = _clip01(0.30 * transition + 0.35 * hazard + 0.20 * (1.0 - conf) + 0.15 * level_risk)
        rows.append(
            {
                "features": [
                    conf,
                    quality,
                    transition,
                    hazard,
                    ews,
                    0.0,
                    0.0,
                    float(age),
                    changed,
                    0.0,
                ],
                "target": target,
                "timestamp": _parse_ts(ts_raw, now),
                "source": "sector_alerts_db",
                "entity": sector_s,
            }
        )
    return rows


def _samples_to_arrays(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not samples:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            [],
        )
    X = np.array([s["features"] for s in samples], dtype=float)
    y = np.array([float(s["target"]) for s in samples], dtype=float)
    ts = np.array([float(s["timestamp"].timestamp()) for s in samples], dtype=float)
    sources = [str(s.get("source", "unknown")) for s in samples]
    return X, y, ts, sources


def _source_counts(sources: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1
    return counts


def _split_temporal_indices(
    timestamps: np.ndarray,
    *,
    holdout_frac: float,
    min_holdout: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(timestamps.shape[0])
    if n <= 12:
        return np.arange(n, dtype=int), np.zeros((0,), dtype=int)

    order = np.argsort(timestamps, kind="stable")
    hold = int(round(max(0.0, holdout_frac) * n))
    hold = max(int(min_holdout), hold)
    hold = min(hold, n - 8)
    if hold <= 0:
        return order, np.zeros((0,), dtype=int)

    split = max(1, n - hold)
    return order[:split], order[split:]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _forward(
    A: np.ndarray,
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    w_out: np.ndarray,
    b_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1 = A @ X @ W1 + b1
    H1 = _relu(Z1)
    Z2 = A @ H1 @ W2 + b2
    H2 = _relu(Z2)
    logits = H2 @ w_out + b_out
    pred = _sigmoid(logits)
    return H1, H2, logits, pred


def _train_gnn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int = 16,
    epochs: int = 600,
    lr: float = 0.01,
    l2: float = 1e-4,
    k_neighbors: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    n, f = X.shape
    if n == 0:
        raise RuntimeError("sem dados para treino do modelo C.")
    mean, std = _norm_stats(X)
    Xz = (X - mean) / std
    A = _build_adjacency(Xz, k=k_neighbors)

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 0.1, size=(f, hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=float)
    W2 = rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim))
    b2 = np.zeros((hidden_dim,), dtype=float)
    w_out = rng.normal(0.0, 0.1, size=(hidden_dim,))
    b_out = 0.0

    losses = []
    for _ in range(epochs):
        H1, H2, _, pred = _forward(A, Xz, W1, b1, W2, b2, w_out, b_out)
        err = pred - y
        loss = float(np.mean(err**2) + l2 * (np.mean(W1**2) + np.mean(W2**2) + np.mean(w_out**2)))
        losses.append(loss)

        n_inv = 1.0 / max(1, n)
        d_pred = 2.0 * err * n_inv
        d_logits = d_pred * pred * (1.0 - pred)

        grad_w_out = H2.T @ d_logits + 2.0 * l2 * w_out
        grad_b_out = float(np.sum(d_logits))
        dH2 = np.outer(d_logits, w_out)

        dZ2 = dH2 * (H2 > 0).astype(float)
        grad_W2 = (A @ H1).T @ dZ2 + 2.0 * l2 * W2
        grad_b2 = np.sum(dZ2, axis=0)
        dH1 = A.T @ (dZ2 @ W2.T)

        dZ1 = dH1 * (H1 > 0).astype(float)
        grad_W1 = (A @ Xz).T @ dZ1 + 2.0 * l2 * W1
        grad_b1 = np.sum(dZ1, axis=0)

        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        w_out -= lr * grad_w_out
        b_out -= lr * grad_b_out

    _, _, _, pred_final = _forward(A, Xz, W1, b1, W2, b2, w_out, b_out)
    rmse = float(np.sqrt(np.mean((pred_final - y) ** 2)))

    return {
        "feature_names": list(FEATURE_NAMES),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "graph": {"k_neighbors": int(k_neighbors)},
        "architecture": {"input_dim": int(f), "hidden_dim": int(hidden_dim)},
        "weights": {
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist(),
            "w_out": w_out.tolist(),
            "b_out": float(b_out),
        },
        "training": {
            "seed": int(seed),
            "epochs": int(epochs),
            "lr": float(lr),
            "l2": float(l2),
            "loss_last": float(losses[-1] if losses else 0.0),
            "loss_best": float(min(losses) if losses else 0.0),
            "rmse": rmse,
            "n_nodes": int(n),
        },
    }


def _infer_with_checkpoint(checkpoint: dict[str, Any], X: np.ndarray) -> dict[str, Any]:
    feat = checkpoint.get("feature_names")
    if not isinstance(feat, list) or len(feat) != X.shape[1]:
        raise RuntimeError("checkpoint feature_names incompativel.")
    norm = checkpoint.get("normalization")
    graph = checkpoint.get("graph")
    w = checkpoint.get("weights")
    if not isinstance(norm, dict) or not isinstance(graph, dict) or not isinstance(w, dict):
        raise RuntimeError("checkpoint incompleto.")

    mean = np.array(norm.get("mean", []), dtype=float)
    std = np.array(norm.get("std", []), dtype=float)
    if mean.shape[0] != X.shape[1] or std.shape[0] != X.shape[1]:
        raise RuntimeError("checkpoint normalization invalido.")
    Xz = (X - mean) / np.where(std < 1e-8, 1.0, std)

    k = int(graph.get("k_neighbors", 4))
    A = _build_adjacency(Xz, k=k)

    W1 = np.array(w.get("W1"), dtype=float)
    b1 = np.array(w.get("b1"), dtype=float)
    W2 = np.array(w.get("W2"), dtype=float)
    b2 = np.array(w.get("b2"), dtype=float)
    w_out = np.array(w.get("w_out"), dtype=float)
    b_out = float(w.get("b_out", 0.0))

    _, _, _, pred = _forward(A, Xz, W1, b1, W2, b2, w_out, b_out)
    return {
        "node_scores": pred.tolist(),
        "risk_score": float(np.clip(np.mean(pred), 0.0, 1.0)),
        "confidence": float(np.clip(1.0 - np.std(pred), 0.0, 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model C GNN checkpoint with expanded panel + temporal holdout.")
    parser.add_argument("--panel", type=str, default="results/validation/risk_truth_panel.json")
    parser.add_argument("--out", type=str, default="models/model_c_gnn_checkpoint.json")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--min-holdout-samples", type=int, default=8)
    parser.add_argument("--disable-extra-panel", action="store_true")
    parser.add_argument("--extra-master-pattern", type=str, default="results/validation/**/master_summary.csv")
    parser.add_argument("--asset-groups-csv", type=str, default="data/asset_groups_470_enriched.csv")
    parser.add_argument("--sector-db", type=str, default="results/event_study_sectors/sector_alerts.db")
    parser.add_argument("--max-extra-samples", type=int, default=5000)
    args = parser.parse_args()

    panel_path = ROOT / args.panel
    panel = _read_json(panel_path, {})
    trained_at = datetime.now(timezone.utc)

    panel_samples = _extract_panel_samples(panel if isinstance(panel, dict) else {}, fallback_ts=trained_at)

    extra_samples: list[dict[str, Any]] = []
    master_paths: list[str] = []
    sector_db_path = ROOT / args.sector_db
    if not bool(args.disable_extra_panel):
        group_map = _load_asset_group_map(ROOT / args.asset_groups_csv)
        for path in sorted(ROOT.glob(args.extra_master_pattern)):
            master_rows = _extract_master_summary_samples(path, asset_groups=group_map)
            if master_rows:
                master_paths.append(_rel_path(path))
                extra_samples.extend(master_rows)
        extra_samples.extend(_extract_sector_db_samples(sector_db_path))

    max_extra = int(max(0, args.max_extra_samples))
    if max_extra > 0 and len(extra_samples) > max_extra:
        extra_samples = sorted(extra_samples, key=lambda x: float(x["timestamp"].timestamp()))[-max_extra:]

    samples = [*panel_samples, *extra_samples]
    X_all, y_all, ts_all, source_all = _samples_to_arrays(samples)
    if X_all.shape[0] == 0:
        raise RuntimeError("risk_truth_panel sem entries para treino.")

    train_idx, holdout_idx = _split_temporal_indices(
        ts_all,
        holdout_frac=float(max(0.0, args.holdout_frac)),
        min_holdout=int(max(0, args.min_holdout_samples)),
    )

    X_train = X_all[train_idx] if train_idx.size else X_all
    y_train = y_all[train_idx] if train_idx.size else y_all
    source_train = [source_all[int(i)] for i in (train_idx.tolist() if train_idx.size else list(range(X_all.shape[0])))]
    source_holdout = [source_all[int(i)] for i in holdout_idx.tolist()] if holdout_idx.size else []

    ckpt = _train_gnn(
        X_train,
        y_train,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        k_neighbors=int(args.k_neighbors),
        seed=int(args.seed),
    )
    infer = _infer_with_checkpoint(ckpt, X_all)

    train_ts = ts_all[train_idx] if train_idx.size else ts_all
    holdout_metrics: dict[str, Any] = {
        "enabled": bool(holdout_idx.size > 0),
        "n_train": int(X_train.shape[0]),
        "n_holdout": int(holdout_idx.size),
        "holdout_frac_requested": float(max(0.0, args.holdout_frac)),
        "start_train_utc": datetime.fromtimestamp(float(np.min(train_ts)), tz=timezone.utc).isoformat(),
        "end_train_utc": datetime.fromtimestamp(float(np.max(train_ts)), tz=timezone.utc).isoformat(),
        "start_holdout_utc": None,
        "end_holdout_utc": None,
        "rmse": None,
        "mae": None,
        "corr": None,
        "baseline_rmse": None,
        "generalization_gap_rmse": None,
    }

    if holdout_idx.size > 0:
        X_hold = X_all[holdout_idx]
        y_hold = y_all[holdout_idx]
        pred_hold = np.array(_infer_with_checkpoint(ckpt, X_hold).get("node_scores", []), dtype=float)
        if pred_hold.shape[0] == y_hold.shape[0] and y_hold.shape[0] > 0:
            rmse = float(np.sqrt(np.mean((pred_hold - y_hold) ** 2)))
            mae = float(np.mean(np.abs(pred_hold - y_hold)))
            baseline = float(np.mean(y_train)) if y_train.size > 0 else 0.5
            baseline_rmse = float(np.sqrt(np.mean((baseline - y_hold) ** 2)))
            corr = None
            if y_hold.size >= 3 and float(np.std(y_hold)) > 1e-8 and float(np.std(pred_hold)) > 1e-8:
                corr = float(np.corrcoef(y_hold, pred_hold)[0, 1])
            holdout_metrics.update(
                {
                    "start_holdout_utc": datetime.fromtimestamp(float(np.min(ts_all[holdout_idx])), tz=timezone.utc).isoformat(),
                    "end_holdout_utc": datetime.fromtimestamp(float(np.max(ts_all[holdout_idx])), tz=timezone.utc).isoformat(),
                    "rmse": rmse,
                    "mae": mae,
                    "corr": corr,
                    "baseline_rmse": baseline_rmse,
                    "generalization_gap_rmse": rmse - float(ckpt["training"]["rmse"]),
                }
            )

    ckpt["training"]["n_nodes_total"] = int(X_all.shape[0])
    ckpt["training"]["n_nodes_train"] = int(X_train.shape[0])
    ckpt["training"]["n_nodes_holdout"] = int(holdout_idx.size)
    ckpt["training"]["source_counts_total"] = _source_counts(source_all)
    ckpt["training"]["source_counts_train"] = _source_counts(source_train)
    ckpt["training"]["source_counts_holdout"] = _source_counts(source_holdout)
    ckpt["training"]["holdout_temporal"] = holdout_metrics

    payload = {
        "version": "model_c_gnn_v1",
        "trained_at_utc": trained_at.isoformat(),
        "source_panel": _rel_path(panel_path),
        "extra_sources": {
            "enabled": not bool(args.disable_extra_panel),
            "extra_master_pattern": str(args.extra_master_pattern),
            "master_summary_paths": master_paths,
            "sector_db": _rel_path(sector_db_path),
            "asset_groups_csv": _rel_path(ROOT / args.asset_groups_csv),
            "max_extra_samples": int(max_extra),
            "n_panel_samples": int(len(panel_samples)),
            "n_extra_samples": int(len(extra_samples)),
        },
        "checkpoint": ckpt,
        "sanity_inference": infer,
        "holdout_temporal": holdout_metrics,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[train_model_c_gnn] out={out_path} n_total={X_all.shape[0]} "
        f"n_train={X_train.shape[0]} n_holdout={holdout_idx.size} "
        f"risk={infer['risk_score']:.3f} conf={infer['confidence']:.3f} "
        f"holdout_rmse={holdout_metrics.get('rmse')}"
    )


if __name__ == "__main__":
    main()
