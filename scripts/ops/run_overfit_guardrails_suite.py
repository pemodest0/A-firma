#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    yt = pd.Series(y_true).dropna().astype(int)
    yp = pd.Series(y_pred).reindex(yt.index).fillna(0).astype(int)
    n = int(yt.shape[0])
    if n <= 0:
        return {
            "n": 0.0,
            "event_rate": float("nan"),
            "alert_rate": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = float((tp + tn) / n)
    return {
        "n": float(n),
        "event_rate": float((yt == 1).mean()),
        "alert_rate": float((yp == 1).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def _random_baseline(y_true: pd.Series, *, alert_rate: float, n_iter: int, seed: int) -> dict[str, float]:
    yt = pd.Series(y_true).dropna().astype(int).to_numpy(dtype=int)
    n = int(yt.size)
    p = float(min(1.0, max(0.0, _safe_float(alert_rate))))
    if n <= 0 or not np.isfinite(p):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    rng = np.random.default_rng(int(seed))
    pre: list[float] = []
    rec: list[float] = []
    f1s: list[float] = []
    for _ in range(int(max(1, n_iter))):
        pred = (rng.random(n) < p).astype(int)
        m = _binary_metrics(pd.Series(yt), pd.Series(pred))
        pre.append(_safe_float(m["precision"]))
        rec.append(_safe_float(m["recall"]))
        f1s.append(_safe_float(m["f1"]))

    def _m(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")

    return {"precision": _m(pre), "recall": _m(rec), "f1": _m(f1s)}


def _safe_lift(metric: float, baseline: float) -> float:
    m = _safe_float(metric)
    b = _safe_float(baseline)
    if not np.isfinite(m) or not np.isfinite(b) or b <= 1e-12:
        return float("nan")
    return float(m / b)


def _latest_impact_dir() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        hier = run / "hierarchical"
        if not hier.exists():
            continue
        for cand in sorted(hier.glob("impact_learning*"), key=lambda p: p.name, reverse=True):
            if not cand.is_dir():
                continue
            needed = [
                cand / "impact_training_dataset.csv",
                cand / "impact_walkforward_monthly_compare.csv",
            ]
            if not all(p.exists() for p in needed):
                continue
            try:
                ds = pd.read_csv(cand / "impact_training_dataset.csv", usecols=["date"], nrows=5)
                cp = pd.read_csv(cand / "impact_walkforward_monthly_compare.csv", usecols=["month"], nrows=5)
            except Exception:
                continue
            if ds.empty or cp.empty:
                continue
            return cand
    raise FileNotFoundError("no impact_learning dir found with required files")


def _latest_yearly_eval_dir() -> Path | None:
    base = ROOT / "results" / "portfolio_sim"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.endswith("_systematic_yearly")], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "systematic_summary.json").exists() and (d / "yearly_systematic_eval.csv").exists() and (d / "monthly_systematic_eval.csv").exists():
            return d
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(path: Path, top_assets_per_day: int) -> pd.DataFrame:
    usecols = [
        "date",
        "asset_id",
        "impact_global",
        "impact_sector",
        "sector_loading",
        "overlap_sector_global",
        "global_share_within_sector",
        "drawdown_label",
        "regime_now_flag",
        "regime_label",
        "global_score",
    ]
    d = pd.read_csv(path, usecols=usecols)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values(["date", "impact_global"], ascending=[True, False]).reset_index(drop=True)
    numeric_cols = [c for c in d.columns if c not in {"date", "asset_id"}]
    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if int(top_assets_per_day) > 0:
        d = d.groupby("date", as_index=False, group_keys=False).head(int(top_assets_per_day)).reset_index(drop=True)
    return d


def _build_masks(df: pd.DataFrame, *, train_end: str, quarantine_start: str) -> dict[str, pd.Series]:
    tr = pd.Timestamp(str(train_end))
    q0 = pd.Timestamp(str(quarantine_start))
    return {
        "train": (df["date"] <= tr),
        "validation": (df["date"] > tr) & (df["date"] < q0),
        "quarantine": (df["date"] >= q0),
    }


def _threshold_from_train(proba: np.ndarray, q: float) -> float:
    arr = np.asarray(proba, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan")
    return float(np.quantile(arr, min(0.999, max(0.001, float(q)))))


def _fit_eval(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    masks: dict[str, pd.Series],
    alert_quantile: float,
    seed: int,
    random_iters: int,
) -> dict[str, Any]:
    needed = ["date", target] + list(features)
    x = df[needed].copy()
    x = x.dropna(subset=["date", target])
    x[target] = pd.to_numeric(x[target], errors="coerce")
    x = x.dropna(subset=[target])
    x[target] = x[target].astype(int)
    x = x.dropna(subset=features)
    if x.empty:
        return {"status": "empty_dataset", "target": str(target), "features": features}

    m_train = masks["train"].reindex(x.index).fillna(False)
    m_val = masks["validation"].reindex(x.index).fillna(False)
    m_qua = masks["quarantine"].reindex(x.index).fillna(False)

    n_train = int(m_train.sum())
    n_val = int(m_val.sum())
    n_qua = int(m_qua.sum())
    if n_train <= 1 or n_val <= 1:
        return {
            "status": "insufficient_split",
            "target": str(target),
            "features": features,
            "n_train": n_train,
            "n_validation": n_val,
            "n_quarantine": n_qua,
        }

    y_train = x.loc[m_train, target].astype(int)
    if y_train.nunique() < 2:
        return {
            "status": "single_class_train",
            "target": str(target),
            "features": features,
            "n_train": n_train,
            "n_validation": n_val,
            "n_quarantine": n_qua,
        }

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=600, random_state=int(seed))),
        ]
    )
    X_train = x.loc[m_train, features]
    model.fit(X_train, y_train)

    proba_train = model.predict_proba(X_train)[:, 1]
    thr = _threshold_from_train(proba_train, q=float(alert_quantile))
    if not np.isfinite(thr):
        return {"status": "threshold_not_finite", "target": str(target), "features": features}

    out: dict[str, Any] = {
        "status": "ok",
        "target": str(target),
        "features": list(features),
        "threshold": float(thr),
        "counts": {
            "train": n_train,
            "validation": n_val,
            "quarantine": n_qua,
        },
        "metrics": {},
    }

    for split_name, split_mask in [("train", m_train), ("validation", m_val), ("quarantine", m_qua)]:
        if int(split_mask.sum()) <= 0:
            out["metrics"][split_name] = {
                "n": 0,
                "event_rate": float("nan"),
                "alert_rate": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "accuracy": float("nan"),
                "random_precision": float("nan"),
                "random_recall": float("nan"),
                "random_f1": float("nan"),
                "lift_precision": float("nan"),
            }
            continue
        ys = x.loc[split_mask, target].astype(int)
        xs = x.loc[split_mask, features]
        ps = model.predict_proba(xs)[:, 1]
        pred = (ps >= float(thr)).astype(int)
        met = _binary_metrics(ys, pd.Series(pred, index=ys.index))
        split_seed_offset = {"train": 101, "validation": 211, "quarantine": 307}.get(split_name, 401)
        target_seed_offset = 1000 if str(target) == "drawdown_label" else 2000
        rb = _random_baseline(
            ys,
            alert_rate=float(met["alert_rate"]),
            n_iter=int(random_iters),
            seed=int(seed) + int(split_seed_offset) + int(target_seed_offset),
        )
        out["metrics"][split_name] = met | {
            "random_precision": _safe_float(rb["precision"]),
            "random_recall": _safe_float(rb["recall"]),
            "random_f1": _safe_float(rb["f1"]),
            "lift_precision": _safe_lift(_safe_float(met["precision"]), _safe_float(rb["precision"])),
        }
    return out


def _fit_auc_eval(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    masks: dict[str, pd.Series],
    seed: int,
) -> dict[str, Any]:
    needed = ["date", target] + list(features)
    x = df[needed].copy()
    x = x.dropna(subset=["date", target])
    x[target] = pd.to_numeric(x[target], errors="coerce")
    x = x.dropna(subset=[target])
    x[target] = x[target].astype(int)
    x = x.dropna(subset=features)
    if x.empty:
        return {"status": "empty_dataset", "target": str(target), "features": features}

    m_train = masks["train"].reindex(x.index).fillna(False)
    m_val = masks["validation"].reindex(x.index).fillna(False)
    m_qua = masks["quarantine"].reindex(x.index).fillna(False)
    if int(m_train.sum()) <= 1 or int(m_val.sum()) <= 1:
        return {
            "status": "insufficient_split",
            "target": str(target),
            "features": features,
            "counts": {"train": int(m_train.sum()), "validation": int(m_val.sum()), "quarantine": int(m_qua.sum())},
        }

    y_train = x.loc[m_train, target].astype(int)
    if y_train.nunique() < 2:
        return {
            "status": "single_class_train",
            "target": str(target),
            "features": features,
            "counts": {"train": int(m_train.sum()), "validation": int(m_val.sum()), "quarantine": int(m_qua.sum())},
        }

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=600, random_state=int(seed))),
        ]
    )
    X_train = x.loc[m_train, features]
    model.fit(X_train, y_train)

    def _split_auc(mask: pd.Series) -> float:
        if int(mask.sum()) <= 1:
            return float("nan")
        ys = x.loc[mask, target].astype(int)
        if ys.nunique() < 2:
            return float("nan")
        xs = x.loc[mask, features]
        ps = model.predict_proba(xs)[:, 1]
        try:
            return float(roc_auc_score(ys, ps))
        except Exception:
            return float("nan")

    return {
        "status": "ok",
        "target": str(target),
        "features": list(features),
        "counts": {"train": int(m_train.sum()), "validation": int(m_val.sum()), "quarantine": int(m_qua.sum())},
        "auc": {
            "train": _split_auc(m_train),
            "validation": _split_auc(m_val),
            "quarantine": _split_auc(m_qua),
        },
    }


def _null_distribution_auc(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    masks: dict[str, pd.Series],
    n_iter: int,
    seed: int,
) -> dict[str, float]:
    x = df[["date", target] + features].copy()
    x = x.dropna(subset=["date", target] + features)
    x[target] = pd.to_numeric(x[target], errors="coerce")
    x = x.dropna(subset=[target])
    x[target] = x[target].astype(int)

    m_train = masks["train"].reindex(x.index).fillna(False)
    m_val = masks["validation"].reindex(x.index).fillna(False)
    if int(m_train.sum()) <= 1 or int(m_val.sum()) <= 1:
        return {"mean_auc": float("nan"), "std_auc": float("nan")}

    y_train = x.loc[m_train, target].astype(int)
    y_val = x.loc[m_val, target].astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return {"mean_auc": float("nan"), "std_auc": float("nan")}

    X_train = x.loc[m_train, features]
    X_val = x.loc[m_val, features]
    rng = np.random.default_rng(int(seed))
    vals: list[float] = []
    for _ in range(int(max(1, n_iter))):
        y_shuffled = y_train.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 10_000_000))).to_numpy(dtype=int)
        if np.unique(y_shuffled).size < 2:
            continue
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=600, random_state=int(seed))),
            ]
        )
        model.fit(X_train, y_shuffled)
        p_val = model.predict_proba(X_val)[:, 1]
        try:
            vals.append(float(roc_auc_score(y_val, p_val)))
        except Exception:
            continue
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean_auc": float("nan"), "std_auc": float("nan")}
    return {"mean_auc": float(arr.mean()), "std_auc": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def _shift_features(df: pd.DataFrame, features: list[str], lag_days: int) -> pd.DataFrame:
    x = df.copy()
    x = x.sort_values(["asset_id", "date"]).reset_index(drop=True)
    for c in features:
        x[c] = x.groupby("asset_id", dropna=False)[c].shift(int(lag_days))
    return x


def _null_distribution(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    masks: dict[str, pd.Series],
    threshold_quantile: float,
    n_iter: int,
    seed: int,
) -> dict[str, float]:
    x = df[["date", target] + features].copy()
    x = x.dropna(subset=["date", target] + features)
    x[target] = pd.to_numeric(x[target], errors="coerce")
    x = x.dropna(subset=[target])
    x[target] = x[target].astype(int)

    m_train = masks["train"].reindex(x.index).fillna(False)
    m_val = masks["validation"].reindex(x.index).fillna(False)
    if int(m_train.sum()) <= 1 or int(m_val.sum()) <= 1:
        return {"mean_f1": float("nan"), "std_f1": float("nan")}

    y_train = x.loc[m_train, target].astype(int)
    if y_train.nunique() < 2:
        return {"mean_f1": float("nan"), "std_f1": float("nan")}

    X_train = x.loc[m_train, features]
    X_val = x.loc[m_val, features]
    y_val = x.loc[m_val, target].astype(int)

    rng = np.random.default_rng(int(seed))
    vals: list[float] = []
    for _ in range(int(max(1, n_iter))):
        y_shuffled = y_train.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 10_000_000))).to_numpy(dtype=int)
        if np.unique(y_shuffled).size < 2:
            continue
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=600, random_state=int(seed))),
            ]
        )
        model.fit(X_train, y_shuffled)
        p_train = model.predict_proba(X_train)[:, 1]
        thr = _threshold_from_train(p_train, q=float(threshold_quantile))
        if not np.isfinite(thr):
            continue
        p_val = model.predict_proba(X_val)[:, 1]
        pred = (p_val >= float(thr)).astype(int)
        vals.append(_safe_float(_binary_metrics(y_val, pd.Series(pred, index=y_val.index))["f1"]))

    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {"mean_f1": float("nan"), "std_f1": float("nan")}
    return {"mean_f1": float(arr.mean()), "std_f1": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def _step2_3_from_walkforward(
    compare_csv: Path,
    *,
    min_months: int,
    max_f1_cv: float,
    min_lift_mean: float,
    min_lift_share: float,
    min_lift_share_ge095: float,
    label_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not compare_csv.exists():
        return {
            "step2_block_stability": {"pass": False, "reason": f"missing_file:{compare_csv}"},
            "step3_baseline_compare": {"pass": False, "reason": f"missing_file:{compare_csv}"},
            "label_summary": [],
        }
    d = pd.read_csv(compare_csv)
    required = {"label", "model", "f1", "lift", "month"}
    if not required.issubset(set(d.columns)):
        return {
            "step2_block_stability": {"pass": False, "reason": "missing_required_columns"},
            "step3_baseline_compare": {"pass": False, "reason": "missing_required_columns"},
            "label_summary": [],
        }

    out_rows: list[dict[str, Any]] = []
    labels = sorted(d["label"].dropna().astype(str).unique().tolist())
    stable_flags: list[bool] = []
    baseline_flags: list[bool] = []
    for lb in labels:
        lb_cfg = (label_thresholds or {}).get(str(lb), {})
        lb_min_lift_mean = float(lb_cfg.get("min_lift_mean", min_lift_mean))
        lb_min_lift_share = float(lb_cfg.get("min_lift_share", min_lift_share))
        lb_min_lift_share_ge095 = float(lb_cfg.get("min_lift_share_ge095", min_lift_share_ge095))

        g = d[d["label"].astype(str) == lb].copy()
        mrows: list[dict[str, Any]] = []
        for model_name, m in g.groupby("model", dropna=False):
            f1 = pd.to_numeric(m["f1"], errors="coerce")
            lift = pd.to_numeric(m["lift"], errors="coerce")
            f1v = f1[np.isfinite(f1)]
            lv = lift[np.isfinite(lift)]
            months = int(pd.Series(m["month"]).astype(str).nunique())
            mean_f1 = float(f1v.mean()) if f1v.shape[0] > 0 else float("nan")
            std_f1 = float(f1v.std(ddof=1)) if f1v.shape[0] > 1 else 0.0
            cv_f1 = float(std_f1 / abs(mean_f1)) if np.isfinite(mean_f1) and abs(mean_f1) > 1e-12 else float("nan")
            mean_lift = float(lv.mean()) if lv.shape[0] > 0 else float("nan")
            share_lift_gt1 = float((lv > 1.0).mean()) if lv.shape[0] > 0 else float("nan")
            share_lift_ge095 = float((lv >= 0.95).mean()) if lv.shape[0] > 0 else float("nan")
            mrows.append(
                {
                    "label": str(lb),
                    "model": str(model_name),
                    "months": months,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                    "cv_f1": cv_f1,
                    "mean_lift": mean_lift,
                    "share_lift_gt1": share_lift_gt1,
                    "share_lift_ge095": share_lift_ge095,
                }
            )
        if not mrows:
            continue
        md = pd.DataFrame(mrows)
        candidate_mask = (
            pd.to_numeric(md["mean_lift"], errors="coerce") >= float(lb_min_lift_mean)
        ) & (
            pd.to_numeric(md["share_lift_gt1"], errors="coerce") >= float(lb_min_lift_share)
        ) & (
            pd.to_numeric(md["share_lift_ge095"], errors="coerce") >= float(lb_min_lift_share_ge095)
        )
        md_sel = md[candidate_mask].copy()
        if md_sel.empty:
            md_sel = md.copy()
        md_sel = md_sel.sort_values(["mean_f1", "mean_lift"], ascending=[False, False]).reset_index(drop=True)
        best = md_sel.iloc[0].to_dict()
        stable_ok = bool(int(best["months"]) >= int(min_months) and np.isfinite(best["cv_f1"]) and float(best["cv_f1"]) <= float(max_f1_cv))
        baseline_ok = bool(
            np.isfinite(best["mean_lift"])
            and np.isfinite(best["share_lift_gt1"])
            and np.isfinite(best["share_lift_ge095"])
            and float(best["mean_lift"]) >= float(lb_min_lift_mean)
            and float(best["share_lift_gt1"]) >= float(lb_min_lift_share)
            and float(best["share_lift_ge095"]) >= float(lb_min_lift_share_ge095)
        )
        stable_flags.append(stable_ok)
        baseline_flags.append(baseline_ok)
        out_rows.append(
            best
            | {
                "stable_ok": stable_ok,
                "baseline_ok": baseline_ok,
                "candidate_filter_applied": bool(candidate_mask.any()),
                "thresholds": {
                    "min_lift_mean": lb_min_lift_mean,
                    "min_lift_share": lb_min_lift_share,
                    "min_lift_share_ge095": lb_min_lift_share_ge095,
                },
            }
        )

    step2 = {"pass": bool(all(stable_flags)) if stable_flags else False, "labels": out_rows}
    step3 = {"pass": bool(all(baseline_flags)) if baseline_flags else False, "labels": out_rows}
    return {"step2_block_stability": step2, "step3_baseline_compare": step3, "label_summary": out_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 8 anti-overfit guardrails and emit promotion decision.")
    ap.add_argument("--impact-dir", type=str, default="")
    ap.add_argument("--yearly-eval-dir", type=str, default="")
    ap.add_argument("--config", type=str, default="config/overfit_guardrails.v1.json")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    cfg = _load_json(ROOT / str(args.config))
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("modeling", {})
    stability_cfg = cfg.get("stability", {})
    quarantine_cfg = cfg.get("quarantine", {})
    gate_cfg = cfg.get("promotion_gate", {})
    advisory_gate_cfg = cfg.get("advisory_gate", {})
    mon_cfg = cfg.get("monitoring", {})

    impact_dir = (ROOT / str(args.impact_dir)).resolve() if str(args.impact_dir).strip() else _latest_impact_dir()
    yearly_eval_dir = (ROOT / str(args.yearly_eval_dir)).resolve() if str(args.yearly_eval_dir).strip() else _latest_yearly_eval_dir()
    outdir = (ROOT / str(args.outdir)).resolve() if str(args.outdir).strip() else (ROOT / "results" / "ops" / "overfit_guardrails" / _run_id())
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_csv = impact_dir / "impact_training_dataset.csv"
    compare_csv = impact_dir / "impact_walkforward_monthly_compare.csv"
    if not dataset_csv.exists():
        raise SystemExit(f"missing dataset: {dataset_csv}")

    df = _load_dataset(dataset_csv, top_assets_per_day=int(split_cfg.get("top_assets_per_day", 80)))
    masks = _build_masks(
        df,
        train_end=str(split_cfg.get("train_end", "2023-12-31")),
        quarantine_start=str(split_cfg.get("quarantine_start", "2025-01-01")),
    )

    # Step 1: temporal split integrity.
    step1_counts = {k: int(v.sum()) for k, v in masks.items()}
    step1_ok = bool(
        step1_counts["train"] >= int(split_cfg.get("min_rows_per_split", 5000))
        and step1_counts["validation"] >= int(split_cfg.get("min_rows_per_split", 5000))
        and step1_counts["quarantine"] >= int(split_cfg.get("min_rows_per_split", 5000))
        and (pd.to_datetime(df.loc[masks["train"], "date"]).max() < pd.to_datetime(df.loc[masks["validation"] | masks["quarantine"], "date"]).min())
    )
    step1 = {
        "pass": step1_ok,
        "counts": step1_counts,
        "train_end": str(split_cfg.get("train_end", "")),
        "quarantine_start": str(split_cfg.get("quarantine_start", "")),
    }

    # Step 2 + Step 3 from walk-forward compare.
    step23 = _step2_3_from_walkforward(
        compare_csv,
        min_months=int(stability_cfg.get("min_months_per_model", 18)),
        max_f1_cv=float(stability_cfg.get("max_f1_cv", 0.9)),
        min_lift_mean=float(stability_cfg.get("min_lift_mean", 1.0)),
        min_lift_share=float(stability_cfg.get("min_lift_share", 0.55)),
        min_lift_share_ge095=float(stability_cfg.get("min_lift_share_ge095", 0.6)),
        label_thresholds=(stability_cfg.get("label_thresholds", {}) if isinstance(stability_cfg, dict) else {}),
    )
    step2 = step23["step2_block_stability"]
    step3 = step23["step3_baseline_compare"]

    # Step 5: complexity control with core vs full.
    core_features = [str(x) for x in model_cfg.get("core_features", ["global_score", "regime_now_flag"])]
    full_features = [str(x) for x in model_cfg.get("full_features", core_features)]
    core_features = [c for c in core_features if c in df.columns]
    full_features = [c for c in full_features if c in df.columns]
    targets = ["drawdown_label", "regime_label"]
    complexity_rows: list[dict[str, Any]] = []
    selected_features_by_target: dict[str, list[str]] = {}
    for target in targets:
        core_eval = _fit_eval(
            df=df,
            target=target,
            features=core_features,
            masks=masks,
            alert_quantile=float(model_cfg.get("alert_quantile", 0.85)),
            seed=int(model_cfg.get("seed", 23)),
            random_iters=int(model_cfg.get("random_iters", 300)),
        )
        full_eval = _fit_eval(
            df=df,
            target=target,
            features=full_features,
            masks=masks,
            alert_quantile=float(model_cfg.get("alert_quantile", 0.85)),
            seed=int(model_cfg.get("seed", 23)),
            random_iters=int(model_cfg.get("random_iters", 300)),
        )
        f1_core = _safe_float(((core_eval.get("metrics") or {}).get("validation") or {}).get("f1"))
        f1_full = _safe_float(((full_eval.get("metrics") or {}).get("validation") or {}).get("f1"))
        gain = _safe_float(f1_full - f1_core)
        min_gain = float(model_cfg.get("min_full_gain_vs_core", 0.01))
        use_full = bool(np.isfinite(gain) and gain >= min_gain)
        selected = full_features if use_full else core_features
        selected_features_by_target[target] = selected
        complexity_rows.append(
            {
                "target": target,
                "f1_validation_core": f1_core,
                "f1_validation_full": f1_full,
                "full_minus_core": gain,
                "min_gain_required": min_gain,
                "selected_feature_set": "full" if use_full else "core",
                "core_status": core_eval.get("status"),
                "full_status": full_eval.get("status"),
            }
        )
    valid_complexity = [
        r
        for r in complexity_rows
        if str(r.get("core_status")) == "ok"
        and str(r.get("full_status")) == "ok"
        and np.isfinite(_safe_float(r.get("f1_validation_core")))
        and np.isfinite(_safe_float(r.get("f1_validation_full")))
    ]
    step5 = {
        "pass": bool(len(valid_complexity) > 0),
        "rows": complexity_rows,
        "selected_features_by_target": selected_features_by_target,
    }

    # Step 6: leakage tests (null + lag diagnostics) on drawdown target using selected set.
    leak_target = "drawdown_label"
    leak_features = selected_features_by_target.get(leak_target, core_features)
    leakage_metric = str(model_cfg.get("leakage_metric", "auc")).strip().lower()
    use_auc = leakage_metric == "auc"

    def _fit_eval_metric(src_df: pd.DataFrame) -> dict[str, Any]:
        if use_auc:
            return _fit_auc_eval(
                df=src_df,
                target=leak_target,
                features=leak_features,
                masks=masks,
                seed=int(model_cfg.get("seed", 23)),
            )
        return _fit_eval(
            df=src_df,
            target=leak_target,
            features=leak_features,
            masks=masks,
            alert_quantile=float(model_cfg.get("alert_quantile", 0.85)),
            seed=int(model_cfg.get("seed", 23)),
            random_iters=int(model_cfg.get("random_iters", 300)),
        )

    def _validation_metric(eval_obj: dict[str, Any]) -> float:
        if use_auc:
            return _safe_float(((eval_obj.get("auc") or {}).get("validation")))
        return _safe_float(((eval_obj.get("metrics") or {}).get("validation") or {}).get("f1"))

    base_eval = _fit_eval_metric(df)
    base_metric_val = _validation_metric(base_eval)
    lag_days = [int(x) for x in model_cfg.get("lag_days", [1, 5])]
    future_gains: list[float] = []
    delay_drops: list[float] = []
    lag_rows: list[dict[str, Any]] = []
    for lag in lag_days:
        d_future = _shift_features(df, leak_features, lag_days=-int(lag))
        fut_eval = _fit_eval_metric(d_future)
        future_metric = _validation_metric(fut_eval)
        d_past = _shift_features(df, leak_features, lag_days=int(lag))
        past_eval = _fit_eval_metric(d_past)
        past_metric = _validation_metric(past_eval)
        future_gain = _safe_float(future_metric - base_metric_val)
        delay_drop = _safe_float(base_metric_val - past_metric)
        future_gains.append(future_gain)
        delay_drops.append(delay_drop)
        lag_rows.append(
            {
                "lag_days": int(lag),
                f"base_{leakage_metric}_validation": base_metric_val,
                f"future_shift_{leakage_metric}_validation": future_metric,
                f"past_shift_{leakage_metric}_validation": past_metric,
                "future_gain_vs_base": future_gain,
                "delay_drop_vs_base": delay_drop,
            }
        )

    if use_auc:
        null_stats = _null_distribution_auc(
            df=df,
            target=leak_target,
            features=leak_features,
            masks=masks,
            n_iter=int(model_cfg.get("null_iters", 40)),
            seed=int(model_cfg.get("seed", 23)),
        )
        null_mean = _safe_float(null_stats.get("mean_auc"))
        null_std = _safe_float(null_stats.get("std_auc"))
    else:
        null_stats = _null_distribution(
            df=df,
            target=leak_target,
            features=leak_features,
            masks=masks,
            threshold_quantile=float(model_cfg.get("alert_quantile", 0.85)),
            n_iter=int(model_cfg.get("null_iters", 40)),
            seed=int(model_cfg.get("seed", 23)),
        )
        null_mean = _safe_float(null_stats.get("mean_f1"))
        null_std = _safe_float(null_stats.get("std_f1"))

    min_null_zscore = float(model_cfg.get("min_null_zscore", 1.0))
    max_future_gain_allowed = float(model_cfg.get("max_lookahead_gain", model_cfg.get("min_lookahead_gain", 0.0005)))
    min_delay_drop = float(model_cfg.get("min_delay_drop_f1", 0.0))
    fg = np.asarray(future_gains, dtype=float)
    fg = fg[np.isfinite(fg)]
    dg = np.asarray(delay_drops, dtype=float)
    dg = dg[np.isfinite(dg)]
    avg_future_gain = _safe_float(fg.mean()) if fg.size > 0 else float("nan")
    avg_delay_drop = _safe_float(dg.mean()) if dg.size > 0 else float("nan")
    null_zscore = float("nan")
    if np.isfinite(base_metric_val) and np.isfinite(null_mean):
        if np.isfinite(null_std) and null_std > 1e-12:
            null_zscore = _safe_float((base_metric_val - null_mean) / null_std)
        elif base_metric_val > null_mean:
            null_zscore = float("inf")
        else:
            null_zscore = float("-inf")
    cond_null = bool(np.isfinite(null_zscore) and null_zscore >= min_null_zscore)
    cond_future = bool(np.isfinite(avg_future_gain) and float(avg_future_gain) <= max_future_gain_allowed)
    cond_delay = bool(np.isfinite(avg_delay_drop) and float(avg_delay_drop) >= min_delay_drop)
    step6 = {
        "pass": bool(cond_null and cond_future and cond_delay),
        "target": leak_target,
        "features_used": leak_features,
        "leakage_metric": leakage_metric,
        f"base_{leakage_metric}_validation": base_metric_val,
        f"null_{leakage_metric}_mean": null_mean,
        f"null_{leakage_metric}_std": null_std,
        "null_zscore": null_zscore,
        "min_null_zscore": min_null_zscore,
        "avg_future_gain": avg_future_gain,
        "max_lookahead_gain_allowed": max_future_gain_allowed,
        "avg_delay_drop": avg_delay_drop,
        "conditions": {
            "cond_null": cond_null,
            "cond_future": cond_future,
            "cond_delay": cond_delay,
        },
        "lag_rows": lag_rows,
    }

    # Step 8: quarantine holdout.
    q_eval = _fit_eval(
        df=df,
        target=leak_target,
        features=leak_features,
        masks=masks,
        alert_quantile=float(model_cfg.get("alert_quantile", 0.85)),
        seed=int(model_cfg.get("seed", 23)),
        random_iters=int(model_cfg.get("random_iters", 300)),
    )
    val_f1 = _safe_float(((q_eval.get("metrics") or {}).get("validation") or {}).get("f1"))
    qua_f1 = _safe_float(((q_eval.get("metrics") or {}).get("quarantine") or {}).get("f1"))
    qua_lift = _safe_float(((q_eval.get("metrics") or {}).get("quarantine") or {}).get("lift_precision"))
    f1_drop = _safe_float(val_f1 - qua_f1)
    max_drop = float(quarantine_cfg.get("max_f1_drop_vs_validation", 0.08))
    min_qua_lift = float(quarantine_cfg.get("min_quarantine_lift", 1.0))
    step8 = {
        "pass": bool(np.isfinite(f1_drop) and f1_drop <= max_drop and np.isfinite(qua_lift) and qua_lift >= min_qua_lift),
        "target": leak_target,
        "validation_f1": val_f1,
        "quarantine_f1": qua_f1,
        "f1_drop_validation_to_quarantine": f1_drop,
        "max_f1_drop_allowed": max_drop,
        "quarantine_lift_precision": qua_lift,
        "min_quarantine_lift_required": min_qua_lift,
    }

    # Step 7: production monitoring health.
    step7: dict[str, Any] = {"pass": False, "reason": "yearly_eval_missing"}
    yearly_summary = {}
    yearly_rows = pd.DataFrame()
    monthly_rows = pd.DataFrame()
    if yearly_eval_dir is not None:
        yearly_summary_path = yearly_eval_dir / "systematic_summary.json"
        yearly_csv = yearly_eval_dir / "yearly_systematic_eval.csv"
        monthly_csv = yearly_eval_dir / "monthly_systematic_eval.csv"
        if yearly_summary_path.exists() and yearly_csv.exists() and monthly_csv.exists():
            yearly_summary = json.loads(yearly_summary_path.read_text(encoding="utf-8"))
            yearly_rows = pd.read_csv(yearly_csv)
            monthly_rows = pd.read_csv(monthly_csv)
            if {"ret", "eqw_ret"}.issubset(set(monthly_rows.columns)):
                alpha = pd.to_numeric(monthly_rows["ret"], errors="coerce") - pd.to_numeric(monthly_rows["eqw_ret"], errors="coerce")
                w = int(mon_cfg.get("rolling_months", 6))
                recent = alpha.dropna().tail(w)
                alpha_mean = _safe_float(recent.mean()) if recent.shape[0] > 0 else float("nan")
                eq = (1.0 + pd.to_numeric(monthly_rows["ret"], errors="coerce").fillna(0.0)).cumprod()
                dd = (eq / eq.cummax()) - 1.0
                max_dd = _safe_float(dd.min()) if dd.shape[0] > 0 else float("nan")
                min_alpha = float(mon_cfg.get("min_alpha_mean_vs_eqw", -0.003))
                dd_floor = float(mon_cfg.get("max_drawdown_floor", -0.35))
                step7 = {
                    "pass": bool(np.isfinite(alpha_mean) and np.isfinite(max_dd) and alpha_mean >= min_alpha and max_dd >= dd_floor),
                    "rolling_months": w,
                    "alpha_mean_recent": alpha_mean,
                    "alpha_min_required": min_alpha,
                    "max_drawdown_strategy": max_dd,
                    "max_drawdown_floor": dd_floor,
                    "source_dir": str(yearly_eval_dir),
                }
            else:
                step7 = {"pass": False, "reason": "monthly_columns_missing", "source_dir": str(yearly_eval_dir)}

    # Step 4: production gate + advisory gate.
    step_map = {
        "step1_temporal_split": bool(step1["pass"]),
        "step2_block_stability": bool(step2["pass"]),
        "step3_baseline_compare": bool(step3["pass"]),
        "step5_complexity_control": bool(step5["pass"]),
        "step6_leakage_tests": bool(step6["pass"]),
        "step7_monitoring": bool(step7.get("pass", False)),
        "step8_quarantine_holdout": bool(step8["pass"]),
    }
    years_tested = int(len(yearly_summary.get("years_tested", []))) if yearly_summary else 0
    worth_rate = _safe_float(yearly_summary.get("worth_it_rate_vs_eqw"))
    prob_pos = _safe_float(yearly_summary.get("monthly_alpha_prob_positive_vs_eqw"))

    def _build_gate(gcfg: dict[str, Any]) -> dict[str, Any]:
        req_steps = [str(x) for x in gcfg.get("required_steps", [])]
        cond_required = bool(all(step_map.get(k, False) for k in req_steps)) if req_steps else True
        cond_years = bool(years_tested >= int(gcfg.get("min_years_tested", 5)))
        cond_worth = bool(np.isfinite(worth_rate) and worth_rate >= float(gcfg.get("min_worth_it_rate_vs_eqw", 0.55)))
        cond_prob = bool(np.isfinite(prob_pos) and prob_pos >= float(gcfg.get("min_prob_positive_vs_eqw", 0.55)))
        return {
            "pass": bool(cond_required and cond_years and cond_worth and cond_prob),
            "conditions": {
                "required_steps_ok": cond_required,
                "years_ok": cond_years,
                "worth_rate_ok": cond_worth,
                "prob_positive_ok": cond_prob,
            },
            "thresholds": {
                "min_years_tested": int(gcfg.get("min_years_tested", 5)),
                "min_worth_it_rate_vs_eqw": float(gcfg.get("min_worth_it_rate_vs_eqw", 0.55)),
                "min_prob_positive_vs_eqw": float(gcfg.get("min_prob_positive_vs_eqw", 0.55)),
                "required_steps": req_steps,
            },
            "observed": {
                "years_tested": years_tested,
                "worth_it_rate_vs_eqw": worth_rate,
                "monthly_alpha_prob_positive_vs_eqw": prob_pos,
                "step_status": step_map,
            },
        }

    step4 = _build_gate(gate_cfg)
    step4_advisory = _build_gate(advisory_gate_cfg)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(ROOT / str(args.config)),
        "impact_dir": str(impact_dir),
        "yearly_eval_dir": str(yearly_eval_dir) if yearly_eval_dir is not None else "",
        "steps": {
            "step1_temporal_split": step1,
            "step2_block_stability": step2,
            "step3_baseline_compare": step3,
            "step4_promotion_gate": step4,
            "step4_advisory_gate": step4_advisory,
            "step5_complexity_control": step5,
            "step6_leakage_tests": step6,
            "step7_monitoring": step7,
            "step8_quarantine_holdout": step8,
        },
        "final_gate": {
            "pass": bool(step4["pass"]),
            "publishable": bool(step4["pass"]),
            "advisory_ready": bool(step4_advisory["pass"]),
        },
    }

    out_json = outdir / "overfit_guardrails_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = []
    for step_name, payload in summary["steps"].items():
        flat_rows.append({"step": step_name, "pass": bool(payload.get("pass", False))})
    steps_csv_path = outdir / "overfit_guardrails_steps.csv"
    pd.DataFrame(flat_rows).to_csv(steps_csv_path, index=False)

    latest_dir = ROOT / "results" / "ops" / "overfit_guardrails" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "overfit_guardrails_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(flat_rows).to_csv(latest_dir / "overfit_guardrails_steps.csv", index=False)

    write_run_manifest(
        outdir,
        script="scripts/ops/run_overfit_guardrails_suite.py",
        params={
            "impact_dir": str(impact_dir),
            "yearly_eval_dir": str(yearly_eval_dir) if yearly_eval_dir is not None else "",
            "config": str(ROOT / str(args.config)),
        },
        paths={
            "summary_json": str(out_json),
            "steps_csv": str(steps_csv_path),
            "latest_summary_json": str(latest_dir / "overfit_guardrails_summary.json"),
            "latest_steps_csv": str(latest_dir / "overfit_guardrails_steps.csv"),
        },
        gates={k: ("ok" if bool(v) else "fail") for k, v in step_map.items()}
        | {
            "promotion_gate": ("ok" if bool(step4["pass"]) else "fail"),
            "advisory_gate": ("ok" if bool(step4_advisory["pass"]) else "fail"),
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "publishable": bool(step4["pass"]),
                "advisory_ready": bool(step4_advisory["pass"]),
                "step_status": step_map,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
