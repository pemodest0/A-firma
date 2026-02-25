#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.ground_truth import (  # noqa: E402
    build_event_label,
    build_regime_future_event_label,
    classification_report_binary,
    threshold_from_train,
)
from engine.structural.impact import (  # noqa: E402
    compute_asset_global_impact,
    compute_asset_sector_impact,
    compute_sector_pair_overlap,
    merge_asset_sector_global_impacts,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _safe_lift(metric: float, baseline: float) -> float:
    m = _safe_float(metric)
    b = _safe_float(baseline)
    if not np.isfinite(m) or not np.isfinite(b) or b <= 1e-12:
        return float("nan")
    return float(m / b)


def _slug_token(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unknown"


def _resolve_backtest_path(run_dir: Path) -> Path:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        official_window = int(float(summary.get("official_window", 0) or 0))
        if official_window > 0:
            p = run_dir / f"backtest_regime_T{official_window}.csv"
            if p.exists():
                return p
    p120 = run_dir / "backtest_regime_T120.csv"
    if p120.exists():
        return p120
    candidates = sorted(run_dir.glob("backtest_regime_T*.csv"))
    if candidates:
        return candidates[-1]
    return run_dir / "backtest_regime_T120.csv"


def _latest_hierarchical_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        g_vec_csv = d / "hierarchical" / "vectors" / "v1_global.csv"
        g_vec_parquet = d / "hierarchical" / "vectors" / "v1_global.parquet"
        if (g_vec_csv.exists() or g_vec_parquet.exists()) and _resolve_backtest_path(d).exists():
            return d
    raise FileNotFoundError("no lab run with hierarchical vectors and backtest found")


def _read_vector(path_no_suffix: Path) -> pd.DataFrame:
    p_parquet = path_no_suffix.with_suffix(".parquet")
    p_csv = path_no_suffix.with_suffix(".csv")
    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            pass
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return pd.DataFrame(columns=["date", "asset_id", "weight"])


def _load_sector_index(hier_dir: Path) -> dict[tuple[str, str], str]:
    idx_path = hier_dir / "universes" / "sector_universe_index.csv"
    out: dict[tuple[str, str], str] = {}
    if not idx_path.exists():
        return out
    df = pd.read_csv(idx_path)
    if df.empty:
        return out
    for _, r in df.iterrows():
        kind = str(r.get("kind", "")).strip().lower()
        slug = str(r.get("slug", "")).strip()
        sector = str(r.get("sector", "")).strip()
        if kind and slug and sector:
            out[(kind, slug)] = sector
    return out


def _load_vectors(run_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    hier_dir = run_dir / "hierarchical"
    vec_dir = hier_dir / "vectors"
    if not vec_dir.exists():
        raise SystemExit(f"missing vectors dir: {vec_dir}")

    global_v1 = _read_vector(vec_dir / "v1_global")
    if global_v1.empty:
        raise SystemExit(f"missing global vectors in: {vec_dir}")

    sector_idx = _load_sector_index(hier_dir)
    sector_map: dict[tuple[str, str], pd.DataFrame] = {}
    for kind in ("gics", "internal"):
        pattern_csv = f"v1_{kind}_*.csv"
        pattern_parquet = f"v1_{kind}_*.parquet"
        files = sorted(list(vec_dir.glob(pattern_csv)) + list(vec_dir.glob(pattern_parquet)))
        seen: set[str] = set()
        for f in files:
            stem = f.stem
            slug = stem.replace(f"v1_{kind}_", "")
            if slug in seen:
                continue
            seen.add(slug)
            sector_name = sector_idx.get((kind, slug), slug)
            raw = _read_vector(vec_dir / f"v1_{kind}_{slug}")
            if not raw.empty:
                sector_map[(kind, str(sector_name))] = raw

    cross_frames: list[pd.DataFrame] = []
    gics_cross = hier_dir / "cross_sector_global_gics_daily.csv"
    if gics_cross.exists():
        d = pd.read_csv(gics_cross)
        if not d.empty:
            d["sector_kind"] = "gics"
            cross_frames.append(d)
    int_cross = hier_dir / "cross_sector_global_internal_daily.csv"
    if int_cross.exists():
        d = pd.read_csv(int_cross)
        if not d.empty:
            d["sector_kind"] = "internal"
            cross_frames.append(d)
    cross_df = pd.concat(cross_frames, ignore_index=True) if cross_frames else pd.DataFrame()
    return global_v1, sector_map, cross_df


def _load_metadata(run_dir: Path) -> pd.DataFrame:
    p1 = run_dir / "hierarchical" / "asset_metadata_used.csv"
    p2 = ROOT / "data" / "asset_metadata.csv"
    if p1.exists():
        return pd.read_csv(p1)
    if p2.exists():
        return pd.read_csv(p2)
    return pd.DataFrame(columns=["asset_id", "ticker"])


def _build_sector_impact_daily(asset_impact: pd.DataFrame) -> pd.DataFrame:
    if asset_impact.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sector_kind",
                "sector",
                "n_assets",
                "sector_loading",
                "overlap_sector_global",
                "impact_global_sum",
                "impact_sector_sum",
                "impact_sector_hhi",
                "dominant_asset",
                "dominant_asset_impact_global",
            ]
        )
    x = asset_impact.copy()
    x["impact_global"] = pd.to_numeric(x["impact_global"], errors="coerce").fillna(0.0)
    x["impact_sector"] = pd.to_numeric(x["impact_sector"], errors="coerce").fillna(0.0)
    x["sector_loading"] = pd.to_numeric(x["sector_loading"], errors="coerce")
    x["overlap_sector_global"] = pd.to_numeric(x["overlap_sector_global"], errors="coerce")
    rows: list[dict[str, Any]] = []
    grp = x.groupby(["date", "sector_kind", "sector"], dropna=False)
    for (d, kind, sec), g in grp:
        gg = g.sort_values("impact_global", ascending=False)
        dom = gg.iloc[0] if not gg.empty else None
        rows.append(
            {
                "date": str(d),
                "sector_kind": str(kind),
                "sector": str(sec),
                "n_assets": int(g["asset_id"].nunique()),
                "sector_loading": _safe_float(g["sector_loading"].dropna().median()),
                "overlap_sector_global": _safe_float(g["overlap_sector_global"].dropna().median()),
                "impact_global_sum": _safe_float(g["impact_global"].sum()),
                "impact_sector_sum": _safe_float(g["impact_sector"].sum()),
                "impact_sector_hhi": _safe_float(np.square(g["impact_sector"]).sum()),
                "dominant_asset": str(dom["asset_id"]) if dom is not None else "",
                "dominant_asset_impact_global": _safe_float(dom["impact_global"]) if dom is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "sector_kind", "sector"]).reset_index(drop=True)


def _build_asset_global_daily(asset_global: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    g = asset_global.copy()
    if g.empty:
        return pd.DataFrame(columns=["date", "asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"])
    g["date"] = g["date"].astype(str)
    g["asset_id"] = g["asset_id"].astype(str)
    g["impact_global"] = pd.to_numeric(g["impact_global"], errors="coerce").fillna(0.0)
    if metadata is not None and (not metadata.empty):
        md = metadata.copy()
        if "asset_id" not in md.columns:
            md["asset_id"] = md.get("ticker", "").astype(str)
        if "ticker" not in md.columns:
            md["ticker"] = md["asset_id"]
        if "sector_gics" not in md.columns:
            md["sector_gics"] = "unknown"
        if "sector_internal" not in md.columns:
            md["sector_internal"] = md["sector_gics"]
        md["asset_id"] = md["asset_id"].astype(str)
        md["ticker"] = md["ticker"].astype(str)
        md["sector_gics"] = md["sector_gics"].astype(str)
        md["sector_internal"] = md["sector_internal"].astype(str)
        g = g.merge(
            md[["asset_id", "ticker", "sector_gics", "sector_internal"]].drop_duplicates(subset=["asset_id"], keep="first"),
            on="asset_id",
            how="left",
        )
    else:
        g["ticker"] = g["asset_id"]
        g["sector_gics"] = "unknown"
        g["sector_internal"] = "unknown"
    return (
        g[["date", "asset_id", "ticker", "impact_global", "sector_gics", "sector_internal"]]
        .sort_values(["date", "asset_id"])
        .reset_index(drop=True)
    )


def _load_returns_core(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "returns_wide_core.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _build_asset_drawdown_labels(
    *,
    returns_core: pd.DataFrame,
    assets: list[str],
    horizon_days: int,
    dd_threshold: float,
) -> pd.DataFrame:
    if returns_core.empty or (not assets):
        return pd.DataFrame(columns=["date", "asset_id", "drawdown_label"])
    rr = returns_core.copy()
    rr["date"] = pd.to_datetime(rr["date"], errors="coerce")
    rr = rr.dropna(subset=["date"]).sort_values("date")
    out_rows: list[pd.DataFrame] = []
    for a in sorted(set([str(x) for x in assets])):
        if a not in rr.columns:
            continue
        r = pd.to_numeric(rr[a], errors="coerce")
        eq = (1.0 + r.fillna(0.0)).cumprod()
        y = build_event_label(equity=eq, horizon_days=int(horizon_days), dd_threshold=float(dd_threshold))
        tmp = pd.DataFrame(
            {
                "date": rr["date"].dt.strftime("%Y-%m-%d"),
                "asset_id": str(a),
                "drawdown_label": pd.to_numeric(y, errors="coerce"),
            }
        )
        out_rows.append(tmp)
    if not out_rows:
        return pd.DataFrame(columns=["date", "asset_id", "drawdown_label"])
    return pd.concat(out_rows, ignore_index=True)


def _build_regime_labels(run_dir: Path, *, horizon_days: int, target_regimes: set[str]) -> pd.DataFrame:
    bt_path = _resolve_backtest_path(run_dir)
    if not bt_path.exists():
        return pd.DataFrame(columns=["date", "regime", "regime_now_flag", "regime_label"])
    bt = pd.read_csv(bt_path)
    if "date" not in bt.columns or "regime" not in bt.columns:
        return pd.DataFrame(columns=["date", "regime", "regime_now_flag", "regime_label"])
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
    bt = bt.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    bt["regime"] = bt["regime"].astype(str).str.lower()
    bt["regime_now_flag"] = bt["regime"].isin(target_regimes).astype(int)
    bt["regime_label"] = build_regime_future_event_label(
        regime=bt["regime"],
        horizon_days=int(horizon_days),
        target_regimes=target_regimes,
    )
    bt["date"] = bt["date"].dt.strftime("%Y-%m-%d")
    return bt[["date", "regime", "regime_now_flag", "regime_label"]].copy()


def _load_global_score(run_dir: Path) -> pd.DataFrame:
    p_h = run_dir / "hierarchical" / "diagnostics_global_score_daily.csv"
    p_g = run_dir / "diagnostics_structural_score_daily.csv"
    p = p_h if p_h.exists() else p_g
    if not p.exists():
        return pd.DataFrame(columns=["date", "global_score"])
    d = pd.read_csv(p)
    if "date" not in d.columns or "score" not in d.columns:
        return pd.DataFrame(columns=["date", "global_score"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d["global_score"] = pd.to_numeric(d["score"], errors="coerce")
    return d[["date", "global_score"]].copy()


def _random_baseline(
    *,
    y_true: pd.Series,
    alert_rate: float,
    seed: int,
    n_iter: int,
) -> dict[str, float]:
    yt = pd.Series(y_true).astype("float")
    y = yt[yt.notna()].astype(int).to_numpy(dtype=int)
    n = int(y.size)
    p = float(min(1.0, max(0.0, _safe_float(alert_rate))))
    if n == 0 or not np.isfinite(p):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    rng = np.random.default_rng(int(seed))
    pre: list[float] = []
    rec: list[float] = []
    f1: list[float] = []
    for _ in range(int(max(1, n_iter))):
        pred = (rng.random(n) < p).astype(int)
        rep = classification_report_binary(y, pred)
        pre.append(_safe_float(rep.get("precision")))
        rec.append(_safe_float(rep.get("recall")))
        f1.append(_safe_float(rep.get("f1")))
    def _mean(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")
    return {"precision": _mean(pre), "recall": _mean(rec), "f1": _mean(f1)}


def _fit_model_probs(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_all: pd.DataFrame,
    seed: int,
    enable_gboost: bool,
    enable_xgboost: bool,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    probs: dict[str, np.ndarray] = {}
    status: dict[str, str] = {}
    y_train_num = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
    if y_train_num.nunique() < 2:
        return probs, {"all": "train_target_single_class"}

    try:
        logit = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=600, random_state=int(seed))),
            ]
        )
        logit.fit(X_train, y_train_num)
        probs["logistic"] = logit.predict_proba(X_all)[:, 1]
        status["logistic"] = "ok"
    except Exception as exc:
        status["logistic"] = f"fail:{exc}"

    if bool(enable_gboost):
        try:
            gbt = GradientBoostingClassifier(random_state=int(seed), n_estimators=250, learning_rate=0.05, max_depth=3)
            gbt.fit(X_train, y_train_num)
            probs["gboost"] = gbt.predict_proba(X_all)[:, 1]
            status["gboost"] = "ok"
        except Exception as exc:
            status["gboost"] = f"fail:{exc}"
    else:
        status["gboost"] = "disabled"

    try:
        lin = LinearRegression()
        lin.fit(X_train, y_train_num.astype(float))
        p_lin = np.asarray(lin.predict(X_all), dtype=float)
        probs["linear"] = np.clip(p_lin, 0.0, 1.0)
        status["linear"] = "ok"
    except Exception as exc:
        status["linear"] = f"fail:{exc}"

    if bool(enable_xgboost):
        try:
            import xgboost as xgb  # type: ignore

            xgb_model = xgb.XGBClassifier(
                random_state=int(seed),
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                n_jobs=1,
                verbosity=0,
            )
            xgb_model.fit(X_train, y_train_num)
            probs["xgboost"] = xgb_model.predict_proba(X_all)[:, 1]
            status["xgboost"] = "ok"
        except Exception as exc:
            status["xgboost"] = f"skip:{exc}"
    else:
        status["xgboost"] = "disabled"

    return probs, status


def _month_starts_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    if pd.isna(start_date) or pd.isna(end_date):
        return []
    s = pd.Timestamp(start_date).normalize().replace(day=1)
    e = pd.Timestamp(end_date).normalize().replace(day=1)
    if s > e:
        return []
    return [pd.Timestamp(x) for x in pd.date_range(start=s, end=e, freq="MS")]


def _normalize_cv_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {"fixed", "expanding"}:
        return m
    return "fixed"


def _build_walkforward_compare_rows(*, wf_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    cols = [
        "month",
        "cv_mode",
        "horizon",
        "label",
        "model",
        "alert_rate",
        "precision",
        "recall",
        "f1",
        "lift",
        "n_events",
    ]
    if wf_df is None or wf_df.empty:
        return pd.DataFrame(columns=cols)
    x = wf_df.copy()
    for c in ["event_rate", "test_rows", "alert_rate", "precision", "recall", "f1", "lift_precision_vs_random"]:
        if c not in x.columns:
            x[c] = np.nan
    x["n_events"] = pd.to_numeric(x["event_rate"], errors="coerce") * pd.to_numeric(x["test_rows"], errors="coerce")
    x["horizon"] = int(max(1, horizon_days))
    if "mode" not in x.columns:
        x["mode"] = ""
    if "target" not in x.columns:
        x["target"] = ""
    x["cv_mode"] = x["mode"].astype(str)
    x["label"] = x["target"].astype(str)
    x["lift"] = pd.to_numeric(x["lift_precision_vs_random"], errors="coerce")
    out = x[
        [
            "month",
            "cv_mode",
            "horizon",
            "label",
            "model",
            "alert_rate",
            "precision",
            "recall",
            "f1",
            "lift",
            "n_events",
        ]
    ].copy()
    out["n_events"] = pd.to_numeric(out["n_events"], errors="coerce").round(6)
    return out.sort_values(["month", "cv_mode", "label", "model"]).reset_index(drop=True)


def _build_label_direction_sanity(ds: pd.DataFrame, *, train_end: str, target_col: str = "drawdown_label") -> dict[str, Any]:
    feature_cols = ["sector_loading", "overlap_sector_global", "global_score"]
    out: dict[str, Any] = {
        "status": "ok",
        "target": str(target_col),
        "threshold_quantile": 0.9,
        "threshold_source": "train",
        "train_end": str(train_end),
        "features": {},
    }
    if ds is None or ds.empty:
        out["status"] = "empty_dataset"
        return out

    x = ds.copy()
    if "date" not in x.columns:
        x["date"] = pd.NaT
    if target_col not in x.columns:
        x[target_col] = np.nan
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[target_col] = pd.to_numeric(x[target_col], errors="coerce")
    x = x.dropna(subset=["date", target_col]).copy()
    if x.empty:
        out["status"] = "empty_target"
        return out
    x[target_col] = x[target_col].astype(int)

    cutoff = pd.Timestamp(str(train_end).strip())
    train_mask = x["date"] <= cutoff
    split_masks = {
        "train": train_mask,
        "test": x["date"] > cutoff,
        "all": pd.Series(True, index=x.index),
    }

    for feat in feature_cols:
        if feat not in x.columns:
            x[feat] = np.nan
        f = pd.to_numeric(x[feat], errors="coerce")
        train_vals = f[train_mask & f.notna()]
        if train_vals.empty:
            out["features"][feat] = {"status": "empty_train_feature"}
            continue
        thr = float(train_vals.quantile(0.90))
        feat_payload: dict[str, Any] = {
            "threshold_top_decile_train": _safe_float(thr),
            "splits": {},
        }
        for split_name, mask in split_masks.items():
            m = mask & f.notna()
            if int(m.sum()) <= 0:
                feat_payload["splits"][split_name] = {
                    "n": 0,
                    "n_high": 0,
                    "event_rate": float("nan"),
                    "event_rate_high": float("nan"),
                    "ratio_high_vs_base": float("nan"),
                }
                continue
            y = x.loc[m, target_col].astype(int)
            ff = f.loc[m].astype(float)
            high = ff >= float(thr)
            p_base = _safe_float(y.mean())
            p_high = _safe_float(y.loc[high].mean()) if int(high.sum()) > 0 else float("nan")
            ratio = _safe_lift(p_high, p_base)
            feat_payload["splits"][split_name] = {
                "n": int(m.sum()),
                "n_high": int(high.sum()),
                "event_rate": p_base,
                "event_rate_high": p_high,
                "ratio_high_vs_base": ratio,
            }
        out["features"][feat] = feat_payload
    return out


def _evaluate_target(
    *,
    ds: pd.DataFrame,
    target_col: str,
    split_date: str,
    alert_quantile: float,
    seed: int,
    random_iters: int,
    enable_gboost: bool,
    enable_xgboost: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = [
        "impact_global",
        "impact_sector",
        "sector_loading",
        "overlap_sector_global",
        "global_share_within_sector",
        "global_score",
    ]
    needed = ["date", target_col, "sector_kind", "sector", "regime_now_flag"] + feature_cols
    for c in needed:
        if c not in ds.columns:
            ds[c] = np.nan

    x = ds.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])
    x[target_col] = pd.to_numeric(x[target_col], errors="coerce")
    x = x.dropna(subset=[target_col]).copy()
    if x.empty:
        return pd.DataFrame(), {"status": "empty_target"}
    x[target_col] = x[target_col].astype(int)

    x["sector_kind"] = x["sector_kind"].astype(str)
    x["sector"] = x["sector"].astype(str)
    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=feature_cols)
    if x.empty:
        return pd.DataFrame(), {"status": "empty_features"}

    cutoff = pd.Timestamp(str(split_date).strip())
    train_mask = x["date"] <= cutoff
    test_mask = x["date"] > cutoff
    if int(train_mask.sum()) < 50 or int(test_mask.sum()) < 30:
        return pd.DataFrame(), {"status": "insufficient_split_rows", "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum())}

    X_num = x[feature_cols].copy()
    X_cat = pd.get_dummies(x[["sector_kind", "sector"]], prefix=["kind", "sector"], drop_first=False, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = x[target_col].astype(int)

    model_probs, model_status = _fit_model_probs(
        X_train=X.loc[train_mask],
        y_train=y.loc[train_mask],
        X_all=X,
        seed=int(seed),
        enable_gboost=bool(enable_gboost),
        enable_xgboost=bool(enable_xgboost),
    )

    rows: list[dict[str, Any]] = []
    for model_name, prob in model_probs.items():
        p = pd.Series(prob, index=x.index).astype(float)
        thr = threshold_from_train(score=p, train_mask=train_mask, q=float(alert_quantile))
        pred = (p >= float(thr)).astype("Int64")
        rep = classification_report_binary(y.loc[test_mask], pred.loc[test_mask])
        rb = _random_baseline(
            y_true=y.loc[test_mask],
            alert_rate=_safe_float(rep.get("alert_rate")),
            seed=int(seed + len(rows) * 1009),
            n_iter=int(random_iters),
        )
        rows.append(
            {
                "target": str(target_col),
                "model": str(model_name),
                "threshold": _safe_float(thr),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "event_rate": _safe_float(rep.get("event_rate")),
                "alert_rate": _safe_float(rep.get("alert_rate")),
                "precision": _safe_float(rep.get("precision")),
                "recall": _safe_float(rep.get("recall")),
                "f1": _safe_float(rep.get("f1")),
                "accuracy": _safe_float(rep.get("accuracy")),
                "random_precision": _safe_float(rb.get("precision")),
                "random_recall": _safe_float(rb.get("recall")),
                "random_f1": _safe_float(rb.get("f1")),
                "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                "status": str(model_status.get(model_name, "ok")),
            }
        )

    if x["global_score"].notna().sum() > 0:
        p = pd.to_numeric(x["global_score"], errors="coerce")
        thr = threshold_from_train(score=p, train_mask=train_mask, q=float(alert_quantile))
        pred = (p >= float(thr)).astype("Int64")
        rep = classification_report_binary(y.loc[test_mask], pred.loc[test_mask])
        rb = _random_baseline(
            y_true=y.loc[test_mask],
            alert_rate=_safe_float(rep.get("alert_rate")),
            seed=int(seed + 41_000),
            n_iter=int(random_iters),
        )
        rows.append(
            {
                "target": str(target_col),
                "model": "score_current",
                "threshold": _safe_float(thr),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "event_rate": _safe_float(rep.get("event_rate")),
                "alert_rate": _safe_float(rep.get("alert_rate")),
                "precision": _safe_float(rep.get("precision")),
                "recall": _safe_float(rep.get("recall")),
                "f1": _safe_float(rep.get("f1")),
                "accuracy": _safe_float(rep.get("accuracy")),
                "random_precision": _safe_float(rb.get("precision")),
                "random_recall": _safe_float(rb.get("recall")),
                "random_f1": _safe_float(rb.get("f1")),
                "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                "status": "ok",
            }
        )

    reg_flag = pd.to_numeric(x["regime_now_flag"], errors="coerce")
    if reg_flag.notna().sum() > 0:
        pred = reg_flag.fillna(0).astype(int).astype("Int64")
        rep = classification_report_binary(y.loc[test_mask], pred.loc[test_mask])
        rb = _random_baseline(
            y_true=y.loc[test_mask],
            alert_rate=_safe_float(rep.get("alert_rate")),
            seed=int(seed + 73_000),
            n_iter=int(random_iters),
        )
        rows.append(
            {
                "target": str(target_col),
                "model": "regime_only",
                "threshold": float("nan"),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "event_rate": _safe_float(rep.get("event_rate")),
                "alert_rate": _safe_float(rep.get("alert_rate")),
                "precision": _safe_float(rep.get("precision")),
                "recall": _safe_float(rep.get("recall")),
                "f1": _safe_float(rep.get("f1")),
                "accuracy": _safe_float(rep.get("accuracy")),
                "random_precision": _safe_float(rb.get("precision")),
                "random_recall": _safe_float(rb.get("recall")),
                "random_f1": _safe_float(rb.get("f1")),
                "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                "status": "ok",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"status": "no_models"}
    best = out.sort_values(["f1", "lift_precision_vs_random"], ascending=[False, False]).iloc[0]
    meta = {
        "status": "ok",
        "split_date": str(cutoff.date()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "event_rate_test": _safe_float(y.loc[test_mask].mean()),
        "best_model": str(best["model"]),
        "best_f1": _safe_float(best["f1"]),
        "best_lift_precision": _safe_float(best["lift_precision_vs_random"]),
    }
    return out, meta


def _evaluate_target_walkforward_monthly(
    *,
    ds: pd.DataFrame,
    target_col: str,
    train_end: str,
    walkforward_start: str,
    walkforward_end: str,
    walkforward_mode: str,
    alert_quantile: float,
    seed: int,
    random_iters: int,
    enable_gboost: bool,
    enable_xgboost: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = [
        "impact_global",
        "impact_sector",
        "sector_loading",
        "overlap_sector_global",
        "global_share_within_sector",
        "global_score",
    ]
    needed = ["date", target_col, "sector_kind", "sector", "regime_now_flag"] + feature_cols
    for c in needed:
        if c not in ds.columns:
            ds[c] = np.nan

    x = ds.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    x[target_col] = pd.to_numeric(x[target_col], errors="coerce")
    x = x.dropna(subset=[target_col]).copy()
    if x.empty:
        return pd.DataFrame(), {"status": "empty_target"}
    x[target_col] = x[target_col].astype(int)
    x["sector_kind"] = x["sector_kind"].astype(str)
    x["sector"] = x["sector"].astype(str)
    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=feature_cols).copy()
    if x.empty:
        return pd.DataFrame(), {"status": "empty_features"}

    X_num = x[feature_cols].copy()
    X_cat = pd.get_dummies(x[["sector_kind", "sector"]], prefix=["kind", "sector"], drop_first=False, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = x[target_col].astype(int)

    train_end_ts = pd.Timestamp(str(train_end).strip())
    wf_start_ts = pd.Timestamp(str(walkforward_start).strip())
    wf_end_ts = pd.Timestamp(str(walkforward_end).strip())
    if wf_start_ts <= train_end_ts:
        wf_start_ts = (train_end_ts + pd.offsets.MonthBegin(1)).normalize()
    wf_end_ts = min(wf_end_ts, pd.Timestamp(x["date"].max()))
    months = _month_starts_between(wf_start_ts, wf_end_ts)
    if not months:
        return pd.DataFrame(), {"status": "empty_month_range", "train_end": str(train_end_ts.date())}

    mode = _normalize_cv_mode(walkforward_mode)

    rows: list[dict[str, Any]] = []
    months_eval = 0
    for i, mstart in enumerate(months):
        mend = pd.Timestamp((mstart + pd.offsets.MonthEnd(0)).date())
        mlabel = str(mstart.strftime("%Y-%m"))
        month_mask = (x["date"] >= mstart) & (x["date"] <= mend)
        if int(month_mask.sum()) <= 0:
            continue

        if mode == "expanding":
            train_cut = pd.Timestamp((mstart - pd.Timedelta(days=1)).date())
        else:
            train_cut = train_end_ts

        train_mask = x["date"] <= train_cut
        if int(train_mask.sum()) < 80:
            continue

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_month = X.loc[month_mask]
        y_month = y.loc[month_mask]
        if int(X_month.shape[0]) < 20:
            continue

        months_eval += 1
        x_all = pd.concat([X_train, X_month], axis=0)
        model_probs, model_status = _fit_model_probs(
            X_train=X_train,
            y_train=y_train,
            X_all=x_all,
            seed=int(seed + i * 13),
            enable_gboost=bool(enable_gboost),
            enable_xgboost=bool(enable_xgboost),
        )

        n_train = int(X_train.shape[0])
        for model_name, p_all in model_probs.items():
            p_all_s = pd.Series(np.asarray(p_all, dtype=float), index=x_all.index)
            p_train = p_all_s.iloc[:n_train]
            p_test = p_all_s.iloc[n_train:]
            thr = threshold_from_train(
                score=p_train,
                train_mask=pd.Series(True, index=p_train.index),
                q=float(alert_quantile),
            )
            pred = (p_test >= float(thr)).astype("Int64")
            rep = classification_report_binary(y_month, pred)
            rb = _random_baseline(
                y_true=y_month,
                alert_rate=_safe_float(rep.get("alert_rate")),
                seed=int(seed + i * 97 + len(rows) * 11),
                n_iter=int(random_iters),
            )
            rows.append(
                {
                    "target": str(target_col),
                    "month": str(mlabel),
                    "month_start": str(mstart.date()),
                    "month_end": str(mend.date()),
                    "train_end_used": str(train_cut.date()),
                    "mode": str(mode),
                    "model": str(model_name),
                    "threshold": _safe_float(thr),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(month_mask.sum()),
                    "event_rate": _safe_float(rep.get("event_rate")),
                    "alert_rate": _safe_float(rep.get("alert_rate")),
                    "precision": _safe_float(rep.get("precision")),
                    "recall": _safe_float(rep.get("recall")),
                    "f1": _safe_float(rep.get("f1")),
                    "accuracy": _safe_float(rep.get("accuracy")),
                    "random_precision": _safe_float(rb.get("precision")),
                    "random_recall": _safe_float(rb.get("recall")),
                    "random_f1": _safe_float(rb.get("f1")),
                    "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                    "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                    "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                    "status": str(model_status.get(model_name, "ok")),
                }
            )

        g_train = pd.to_numeric(x.loc[train_mask, "global_score"], errors="coerce")
        g_test = pd.to_numeric(x.loc[month_mask, "global_score"], errors="coerce")
        if g_train.notna().sum() > 0 and g_test.notna().sum() > 0:
            thr = threshold_from_train(
                score=g_train,
                train_mask=pd.Series(True, index=g_train.index),
                q=float(alert_quantile),
            )
            pred = (g_test >= float(thr)).astype("Int64")
            rep = classification_report_binary(y_month, pred)
            rb = _random_baseline(
                y_true=y_month,
                alert_rate=_safe_float(rep.get("alert_rate")),
                seed=int(seed + i * 101 + 41_000),
                n_iter=int(random_iters),
            )
            rows.append(
                {
                    "target": str(target_col),
                    "month": str(mlabel),
                    "month_start": str(mstart.date()),
                    "month_end": str(mend.date()),
                    "train_end_used": str(train_cut.date()),
                    "mode": str(mode),
                    "model": "score_current",
                    "threshold": _safe_float(thr),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(month_mask.sum()),
                    "event_rate": _safe_float(rep.get("event_rate")),
                    "alert_rate": _safe_float(rep.get("alert_rate")),
                    "precision": _safe_float(rep.get("precision")),
                    "recall": _safe_float(rep.get("recall")),
                    "f1": _safe_float(rep.get("f1")),
                    "accuracy": _safe_float(rep.get("accuracy")),
                    "random_precision": _safe_float(rb.get("precision")),
                    "random_recall": _safe_float(rb.get("recall")),
                    "random_f1": _safe_float(rb.get("f1")),
                    "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                    "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                    "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                    "status": "ok",
                }
            )

        reg_flag = pd.to_numeric(x.loc[month_mask, "regime_now_flag"], errors="coerce")
        if reg_flag.notna().sum() > 0:
            pred = reg_flag.fillna(0).astype(int).astype("Int64")
            rep = classification_report_binary(y_month, pred)
            rb = _random_baseline(
                y_true=y_month,
                alert_rate=_safe_float(rep.get("alert_rate")),
                seed=int(seed + i * 107 + 73_000),
                n_iter=int(random_iters),
            )
            rows.append(
                {
                    "target": str(target_col),
                    "month": str(mlabel),
                    "month_start": str(mstart.date()),
                    "month_end": str(mend.date()),
                    "train_end_used": str(train_cut.date()),
                    "mode": str(mode),
                    "model": "regime_only",
                    "threshold": float("nan"),
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(month_mask.sum()),
                    "event_rate": _safe_float(rep.get("event_rate")),
                    "alert_rate": _safe_float(rep.get("alert_rate")),
                    "precision": _safe_float(rep.get("precision")),
                    "recall": _safe_float(rep.get("recall")),
                    "f1": _safe_float(rep.get("f1")),
                    "accuracy": _safe_float(rep.get("accuracy")),
                    "random_precision": _safe_float(rb.get("precision")),
                    "random_recall": _safe_float(rb.get("recall")),
                    "random_f1": _safe_float(rb.get("f1")),
                    "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                    "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                    "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                    "status": "ok",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {
            "status": "no_months_evaluated",
            "train_end": str(train_end_ts.date()),
            "walkforward_start": str(wf_start_ts.date()),
            "walkforward_end": str(wf_end_ts.date()),
            "mode": str(mode),
            "months_total": int(len(months)),
            "months_evaluated": int(months_eval),
        }
    agg = (
        out.groupby("model", dropna=False)
        .agg(
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_lift_precision=("lift_precision_vs_random", "mean"),
            months=("month", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_lift_precision"], ascending=[False, False])
    )
    top = agg.iloc[0]
    meta = {
        "status": "ok",
        "mode": str(mode),
        "train_end": str(train_end_ts.date()),
        "walkforward_start": str(wf_start_ts.date()),
        "walkforward_end": str(wf_end_ts.date()),
        "months_total": int(len(months)),
        "months_evaluated": int(months_eval),
        "best_model_by_mean_f1": str(top["model"]),
        "best_model_mean_f1": _safe_float(top["mean_f1"]),
        "best_model_mean_lift_precision": _safe_float(top["mean_lift_precision"]),
    }
    return out, meta


def _build_dependency_relations(ds: pd.DataFrame, *, train_end: str) -> pd.DataFrame:
    if ds.empty:
        return pd.DataFrame(columns=["split", "target", "feature", "n", "spearman_corr", "pearson_corr"])
    x = ds.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).copy()
    feature_cols = [
        "impact_global",
        "impact_sector",
        "sector_loading",
        "overlap_sector_global",
        "global_share_within_sector",
        "global_score",
    ]
    for c in feature_cols + ["drawdown_label", "regime_label"]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    cutoff = pd.Timestamp(str(train_end).strip())
    splits = {
        "train": x["date"] <= cutoff,
        "test": x["date"] > cutoff,
        "all": pd.Series(True, index=x.index),
    }
    rows: list[dict[str, Any]] = []
    for split_name, mask in splits.items():
        xx = x.loc[mask].copy()
        if xx.empty:
            continue
        for target in ["drawdown_label", "regime_label"]:
            y = pd.to_numeric(xx[target], errors="coerce")
            for feat in feature_cols:
                f = pd.to_numeric(xx[feat], errors="coerce")
                m = y.notna() & f.notna()
                n = int(m.sum())
                if n < 30:
                    rows.append(
                        {
                            "split": str(split_name),
                            "target": str(target),
                            "feature": str(feat),
                            "n": n,
                            "spearman_corr": float("nan"),
                            "pearson_corr": float("nan"),
                        }
                    )
                    continue
                yy = y[m].astype(float)
                ff = f[m].astype(float)
                spe = float(ff.corr(yy, method="spearman"))
                pea = float(ff.corr(yy, method="pearson"))
                rows.append(
                    {
                        "split": str(split_name),
                        "target": str(target),
                        "feature": str(feat),
                        "n": n,
                        "spearman_corr": _safe_float(spe),
                        "pearson_corr": _safe_float(pea),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["split", "target", "feature", "n", "spearman_corr", "pearson_corr"])
    return pd.DataFrame(rows).sort_values(["split", "target", "feature"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build structural impact tables and evaluate learnability for IA.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--horizon-days", type=int, default=10)
    ap.add_argument("--drawdown-threshold", type=float, default=0.05)
    ap.add_argument("--target-regimes", type=str, default="stress,transition")
    ap.add_argument("--split-date", type=str, default="2024-12-31")
    ap.add_argument("--train-end", type=str, default="2023-12-31")
    ap.add_argument("--walkforward-monthly", type=int, default=0, help="Run monthly walk-forward evaluation after train-end.")
    ap.add_argument("--walkforward-start", type=str, default="2024-01-01")
    ap.add_argument("--walkforward-end", type=str, default="")
    ap.add_argument("--cv-mode", type=str, default="fixed", choices=["fixed", "expanding"])
    ap.add_argument(
        "--walkforward-mode",
        type=str,
        default="",
        help="Deprecated alias for --cv-mode (fixed|expanding).",
    )
    ap.add_argument(
        "--walkforward-compare",
        type=int,
        default=0,
        help="If 1, run walk-forward for both fixed and expanding and export compare CSV.",
    )
    ap.add_argument("--enable-gboost", type=int, default=1, help="Enable GradientBoosting model.")
    ap.add_argument("--enable-xgboost", type=int, default=0, help="Enable XGBoost model (heavier).")
    ap.add_argument("--alert-quantile", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--random-iters", type=int, default=300)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_hierarchical_run()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (run_dir / "hierarchical" / "impact_learning")
    outdir.mkdir(parents=True, exist_ok=True)

    target_regimes = {x.strip().lower() for x in str(args.target_regimes).split(",") if x.strip()}
    if not target_regimes:
        target_regimes = {"stress", "transition"}
    cv_mode = _normalize_cv_mode(str(args.walkforward_mode).strip()) if str(args.walkforward_mode).strip() else _normalize_cv_mode(args.cv_mode)
    compare_modes_enabled = bool(int(args.walkforward_compare))

    global_v1, sector_vectors, cross_df = _load_vectors(run_dir)
    metadata = _load_metadata(run_dir)

    asset_global = compute_asset_global_impact(global_v1)
    asset_sector = compute_asset_sector_impact(sector_vectors)
    asset_impact = merge_asset_sector_global_impacts(
        asset_global=asset_global,
        asset_sector=asset_sector,
        cross_daily=cross_df,
        metadata=metadata,
    )
    asset_global_daily = _build_asset_global_daily(asset_global=asset_global, metadata=metadata)
    sector_impact = _build_sector_impact_daily(asset_impact)
    sector_pair = compute_sector_pair_overlap(sector_vectors)

    returns_core = _load_returns_core(run_dir)
    draw_labels = _build_asset_drawdown_labels(
        returns_core=returns_core,
        assets=asset_impact["asset_id"].astype(str).unique().tolist() if not asset_impact.empty else [],
        horizon_days=int(max(1, args.horizon_days)),
        dd_threshold=float(args.drawdown_threshold),
    )
    regime_labels = _build_regime_labels(
        run_dir=run_dir,
        horizon_days=int(max(1, args.horizon_days)),
        target_regimes=target_regimes,
    )
    global_score = _load_global_score(run_dir)

    dataset = asset_impact.copy()
    dataset = dataset.merge(draw_labels, on=["date", "asset_id"], how="left")
    dataset = dataset.merge(regime_labels, on=["date"], how="left")
    dataset = dataset.merge(global_score, on=["date"], how="left")
    dep_df = _build_dependency_relations(dataset, train_end=str(args.train_end))

    eval_frames: list[pd.DataFrame] = []
    eval_meta: dict[str, Any] = {}
    for target in ["drawdown_label", "regime_label"]:
        ev, meta = _evaluate_target(
            ds=dataset,
            target_col=target,
            split_date=str(args.split_date),
            alert_quantile=float(args.alert_quantile),
            seed=int(args.seed),
            random_iters=int(args.random_iters),
            enable_gboost=bool(int(args.enable_gboost)),
            enable_xgboost=bool(int(args.enable_xgboost)),
        )
        if not ev.empty:
            eval_frames.append(ev)
        eval_meta[target] = meta
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()

    wf_eval_frames: list[pd.DataFrame] = []
    wf_eval_meta: dict[str, Any] = {}
    walkforward_enabled = bool(int(args.walkforward_monthly))
    if str(args.walkforward_end).strip():
        wf_end = str(args.walkforward_end).strip()
    else:
        dmax = pd.to_datetime(dataset.get("date"), errors="coerce").max() if not dataset.empty else pd.NaT
        wf_end = str(dmax.date()) if pd.notna(dmax) else str(args.train_end)
    if walkforward_enabled:
        modes_to_run = [str(cv_mode)]
        if compare_modes_enabled:
            modes_to_run = ["fixed", "expanding"]
        for target in ["drawdown_label", "regime_label"]:
            mode_meta: dict[str, Any] = {}
            for mode in modes_to_run:
                wf_df, wf_meta = _evaluate_target_walkforward_monthly(
                    ds=dataset,
                    target_col=target,
                    train_end=str(args.train_end),
                    walkforward_start=str(args.walkforward_start),
                    walkforward_end=str(wf_end),
                    walkforward_mode=str(mode),
                    alert_quantile=float(args.alert_quantile),
                    seed=int(args.seed),
                    random_iters=int(args.random_iters),
                    enable_gboost=bool(int(args.enable_gboost)),
                    enable_xgboost=bool(int(args.enable_xgboost)),
                )
                if not wf_df.empty:
                    wf_eval_frames.append(wf_df)
                mode_meta[str(mode)] = wf_meta
            if compare_modes_enabled:
                wf_eval_meta[target] = {"selected_mode": str(cv_mode), "modes": mode_meta}
            else:
                wf_eval_meta[target] = mode_meta.get(str(cv_mode), {"status": "no_mode_meta"})
    wf_eval_df = pd.concat(wf_eval_frames, ignore_index=True) if wf_eval_frames else pd.DataFrame()
    wf_compare_df = _build_walkforward_compare_rows(wf_df=wf_eval_df, horizon_days=int(max(1, args.horizon_days)))
    label_direction_sanity = _build_label_direction_sanity(dataset, train_end=str(args.train_end), target_col="drawdown_label")

    asset_csv = outdir / "asset_impact_daily.csv"
    asset_global_csv = outdir / "asset_global_impact_daily.csv"
    sector_csv = outdir / "sector_impact_daily.csv"
    pair_csv = outdir / "sector_pair_overlap_daily.csv"
    dataset_csv = outdir / "impact_training_dataset.csv"
    dep_csv = outdir / "impact_dependency_relations.csv"
    eval_csv = outdir / "impact_model_eval.csv"
    wf_eval_csv = outdir / "impact_walkforward_monthly.csv"
    wf_compare_csv = outdir / "impact_walkforward_monthly_compare.csv"
    sanity_json = outdir / "impact_label_direction_sanity.json"

    asset_global_daily.to_csv(asset_global_csv, index=False)
    asset_impact.to_csv(asset_csv, index=False)
    sector_impact.to_csv(sector_csv, index=False)
    sector_pair.to_csv(pair_csv, index=False)
    dataset.to_csv(dataset_csv, index=False)
    dep_df.to_csv(dep_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    wf_eval_df.to_csv(wf_eval_csv, index=False)
    wf_compare_df.to_csv(wf_compare_csv, index=False)
    sanity_json.write_text(json.dumps(label_direction_sanity, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "params": {
            "horizon_days": int(max(1, args.horizon_days)),
            "drawdown_threshold": float(args.drawdown_threshold),
            "target_regimes": sorted(target_regimes),
            "split_date": str(args.split_date),
            "train_end": str(args.train_end),
            "walkforward_monthly": bool(int(args.walkforward_monthly)),
            "walkforward_start": str(args.walkforward_start),
            "walkforward_end": str(wf_end),
            "cv_mode": str(cv_mode),
            "walkforward_compare": bool(compare_modes_enabled),
            "enable_gboost": bool(int(args.enable_gboost)),
            "enable_xgboost": bool(int(args.enable_xgboost)),
            "alert_quantile": float(args.alert_quantile),
            "seed": int(args.seed),
            "random_iters": int(args.random_iters),
        },
        "counts": {
            "asset_global_rows": int(asset_global_daily.shape[0]),
            "asset_impact_rows": int(asset_impact.shape[0]),
            "sector_impact_rows": int(sector_impact.shape[0]),
            "sector_pair_rows": int(sector_pair.shape[0]),
            "dataset_rows": int(dataset.shape[0]),
            "dependency_rows": int(dep_df.shape[0]),
            "eval_rows": int(eval_df.shape[0]),
            "walkforward_rows": int(wf_eval_df.shape[0]),
            "walkforward_compare_rows": int(wf_compare_df.shape[0]),
            "n_assets": int(asset_impact["asset_id"].nunique()) if not asset_impact.empty else 0,
            "n_sectors": int(sector_impact[["sector_kind", "sector"]].drop_duplicates().shape[0]) if not sector_impact.empty else 0,
        },
        "evaluation": eval_meta,
        "walkforward_evaluation": wf_eval_meta,
        "label_direction_sanity": label_direction_sanity,
        "files": {
            "asset_global_impact_daily_csv": str(asset_global_csv),
            "asset_impact_daily_csv": str(asset_csv),
            "sector_impact_daily_csv": str(sector_csv),
            "sector_pair_overlap_daily_csv": str(pair_csv),
            "impact_training_dataset_csv": str(dataset_csv),
            "impact_dependency_relations_csv": str(dep_csv),
            "impact_model_eval_csv": str(eval_csv),
            "impact_walkforward_monthly_csv": str(wf_eval_csv),
            "impact_walkforward_monthly_compare_csv": str(wf_compare_csv),
            "impact_label_direction_sanity_json": str(sanity_json),
            "impact_summary_json": str(outdir / "impact_summary.json"),
        },
    }
    summary_path = outdir / "impact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_dir = ROOT / "results" / "ops" / "ai_knowledge"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / "latest_structural_impact.json"
    latest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at_utc": summary["generated_at_utc"],
                "source_run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "asset_impact_csv": str(asset_csv),
                "asset_global_impact_csv": str(asset_global_csv),
                "eval_csv": str(eval_csv),
                "walkforward_csv": str(wf_eval_csv),
                "walkforward_compare_csv": str(wf_compare_csv),
                "label_direction_sanity_json": str(sanity_json),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_run_manifest(
        outdir,
        script="scripts/structural/run_structural_impact_learning.py",
        params=summary["params"],
        paths=summary["files"] | {"latest_structural_impact_json": str(latest_path)},
        gates={
            "asset_global_nonempty": bool(asset_global_daily.shape[0] > 0),
            "asset_impact_nonempty": bool(asset_impact.shape[0] > 0),
            "dataset_nonempty": bool(dataset.shape[0] > 0),
            "walkforward_written": bool(wf_eval_csv.exists()),
            "walkforward_compare_written": bool(wf_compare_csv.exists()),
            "summary_written": bool(summary_path.exists()),
            "label_direction_sanity_written": bool(sanity_json.exists()),
        },
        extra={
            "run_dir": str(run_dir),
            "n_sector_vectors": int(len(sector_vectors)),
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "asset_rows": int(asset_impact.shape[0]),
                "eval_rows": int(eval_df.shape[0]),
                "walkforward_rows": int(wf_eval_df.shape[0]),
                "walkforward_compare_rows": int(wf_compare_df.shape[0]),
                "best_drawdown": eval_meta.get("drawdown_label", {}),
                "best_regime": eval_meta.get("regime_label", {}),
                "walkforward_drawdown": wf_eval_meta.get("drawdown_label", {}),
                "walkforward_regime": wf_eval_meta.get("regime_label", {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
