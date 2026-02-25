#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.ground_truth import (  # noqa: E402
    build_event_label,
    build_regime_future_event_label,
    classification_report_binary,
    threshold_from_train,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _safe_lift(metric: float, baseline: float) -> float:
    m = _safe_float(metric)
    b = _safe_float(baseline)
    if (not np.isfinite(m)) or (not np.isfinite(b)) or b <= 1e-12:
        return float("nan")
    return float(m / b)


def _parse_int_list(text: str, *, default: list[int]) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    out = sorted(set(max(1, int(x)) for x in out))
    return out or default


def _parse_float_list(text: str, *, default: list[float]) -> list[float]:
    out: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        x = float(token)
        out.append(min(0.999, max(0.001, x)))
    out = sorted(set(out))
    return out or default


def _parse_date_list(text: str, *, default: list[str]) -> list[pd.Timestamp]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    vals = vals or default
    return [pd.Timestamp(x) for x in vals]


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


def _latest_lab_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        has_global = (d / "diagnostics_structural_score_daily.csv").exists()
        has_hier_global = (d / "hierarchical" / "diagnostics_global_score_daily.csv").exists()
        if (has_global or has_hier_global) and _resolve_backtest_path(d).exists():
            return d
    raise FileNotFoundError("no run with diagnostics score + backtest file found")


def _random_baseline(
    y_true: pd.Series,
    *,
    alert_rate: float,
    n_iter: int,
    seed: int,
) -> dict[str, float]:
    yt = pd.Series(y_true).astype("float")
    y = yt[yt.notna()].astype(int).to_numpy(dtype=int)
    n = int(y.size)
    p = float(min(1.0, max(0.0, _safe_float(alert_rate))))
    if n == 0 or (not np.isfinite(p)):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    rng = np.random.default_rng(int(seed))
    prec: list[float] = []
    rec: list[float] = []
    f1s: list[float] = []
    for _ in range(int(max(1, n_iter))):
        pred = (rng.random(n) < p).astype(int)
        rep = classification_report_binary(y, pred)
        prec.append(_safe_float(rep.get("precision")))
        rec.append(_safe_float(rep.get("recall")))
        f1s.append(_safe_float(rep.get("f1")))

    def _mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")

    return {"precision": _mean(prec), "recall": _mean(rec), "f1": _mean(f1s)}


def _load_global_reference(run_dir: Path) -> pd.DataFrame:
    bt_path = _resolve_backtest_path(run_dir)
    if not bt_path.exists():
        raise FileNotFoundError(f"missing backtest file: {bt_path}")
    bt_df = pd.read_csv(bt_path)
    need_bt = {"date", "regime", "benchmark_equity"}
    if not need_bt.issubset(set(bt_df.columns)):
        miss = sorted(need_bt - set(bt_df.columns))
        raise SystemExit(f"missing columns in backtest file: {miss}")
    bt_df["date"] = pd.to_datetime(bt_df["date"], errors="coerce")
    bt_df["benchmark_equity"] = pd.to_numeric(bt_df["benchmark_equity"], errors="coerce")
    return bt_df[["date", "regime", "benchmark_equity"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _load_returns_core(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "returns_wide_core.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _load_universe_assets(path: Path, returns_cols: set[str]) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    out: list[str] = []
    for col in ["asset_id", "ticker"]:
        if col not in df.columns:
            continue
        vals = [str(x).strip() for x in df[col].tolist()]
        out.extend([x for x in vals if x and (x in returns_cols)])
    out = sorted(set(out))
    return out


def _build_equity_from_returns(returns_core: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    if returns_core.empty or (not assets):
        return pd.DataFrame(columns=["date", "benchmark_equity"])
    cols = [c for c in assets if c in returns_core.columns]
    if not cols:
        return pd.DataFrame(columns=["date", "benchmark_equity"])
    block = returns_core[cols].copy()
    valid_count = block.notna().sum(axis=1)
    mean_ret = block.mean(axis=1, skipna=True)
    mean_ret[valid_count <= 0] = np.nan
    equity = (1.0 + mean_ret.fillna(0.0)).cumprod()
    equity[valid_count <= 0] = np.nan
    out = pd.DataFrame({"date": pd.DatetimeIndex(returns_core.index), "benchmark_equity": equity.values})
    return out


def _load_score(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "date" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["score"] = pd.to_numeric(x["score"], errors="coerce")
    x["flags_valid"] = pd.to_numeric(x.get("flags_valid", 1), errors="coerce").fillna(0).astype(int)
    x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return x


def _discover_universes(run_dir: Path, selector: str) -> list[dict[str, Any]]:
    hier_dir = run_dir / "hierarchical"
    universes: list[dict[str, Any]] = []
    global_hier = hier_dir / "diagnostics_global_score_daily.csv"
    global_root = run_dir / "diagnostics_structural_score_daily.csv"
    if global_hier.exists():
        universes.append(
            {
                "name": "global",
                "kind": "global",
                "sector": "global",
                "slug": "global",
                "score_path": global_hier,
                "assets_path": hier_dir / "universes" / "global_universe.csv",
            }
        )
    elif global_root.exists():
        universes.append(
            {
                "name": "global",
                "kind": "global",
                "sector": "global",
                "slug": "global",
                "score_path": global_root,
                "assets_path": Path(),
            }
        )

    idx_path = hier_dir / "universes" / "sector_universe_index.csv"
    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        if not idx.empty:
            idx = idx.sort_values(["kind", "sector", "slug"]).reset_index(drop=True)
            for _, r in idx.iterrows():
                kind = str(r.get("kind", "")).strip().lower()
                sec = str(r.get("sector", "")).strip()
                slug = str(r.get("slug", "")).strip()
                if kind not in {"gics", "internal"} or (not sec) or (not slug):
                    continue
                score_path = hier_dir / f"diagnostics_sector_{kind}_{slug}_score_daily.csv"
                assets_path = hier_dir / "universes" / f"sector_universe_{kind}_{slug}.csv"
                if score_path.exists():
                    universes.append(
                        {
                            "name": f"{kind}:{sec}",
                            "kind": kind,
                            "sector": sec,
                            "slug": slug,
                            "score_path": score_path,
                            "assets_path": assets_path,
                        }
                    )
    else:
        for p in sorted(hier_dir.glob("diagnostics_sector_gics_*_score_daily.csv")):
            slug = p.stem.replace("diagnostics_sector_gics_", "").replace("_score_daily", "")
            universes.append(
                {
                    "name": f"gics:{slug}",
                    "kind": "gics",
                    "sector": slug,
                    "slug": slug,
                    "score_path": p,
                    "assets_path": hier_dir / "universes" / f"sector_universe_gics_{slug}.csv",
                }
            )
        for p in sorted(hier_dir.glob("diagnostics_sector_internal_*_score_daily.csv")):
            slug = p.stem.replace("diagnostics_sector_internal_", "").replace("_score_daily", "")
            universes.append(
                {
                    "name": f"internal:{slug}",
                    "kind": "internal",
                    "sector": slug,
                    "slug": slug,
                    "score_path": p,
                    "assets_path": hier_dir / "universes" / f"sector_universe_internal_{slug}.csv",
                }
            )

    sel = str(selector).strip().lower() or "global"
    if sel == "all":
        return universes
    if sel == "global":
        return [u for u in universes if u["name"] == "global"]

    if ":" not in sel:
        return []
    skind, sval = sel.split(":", 1)
    skind = skind.strip()
    sval = sval.strip()
    sval_slug = _slug_token(sval)
    out = []
    for u in universes:
        if u["kind"] != skind:
            continue
        sec_norm = str(u["sector"]).strip().lower()
        slug_norm = str(u["slug"]).strip().lower()
        if sec_norm == sval.lower() or slug_norm == sval_slug:
            out.append(u)
    return out


def _merge_universe_data(
    *,
    universe: dict[str, Any],
    global_ref: pd.DataFrame,
    returns_core: pd.DataFrame,
) -> pd.DataFrame:
    score_df = _load_score(Path(universe["score_path"]))
    if score_df.empty:
        return pd.DataFrame()

    if universe["name"] == "global":
        merged = score_df.merge(global_ref, on="date", how="inner")
        merged = merged.rename(columns={"benchmark_equity": "equity"})
        return merged.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)

    returns_cols = {str(c) for c in returns_core.columns}
    assets_path = Path(universe.get("assets_path", ""))
    assets = _load_universe_assets(assets_path, returns_cols=returns_cols) if assets_path else []
    eq_df = _build_equity_from_returns(returns_core=returns_core, assets=assets)
    if eq_df.empty:
        return pd.DataFrame()
    merged = score_df.merge(eq_df, on="date", how="inner")
    merged = merged.rename(columns={"benchmark_equity": "equity"})
    if merged.empty:
        return pd.DataFrame()
    return merged.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)


def _median_lead_time_drawdown(
    *,
    equity: pd.Series,
    pred: pd.Series,
    valid_mask: pd.Series,
    horizon: int,
    dd_threshold: float,
) -> float:
    eq = pd.to_numeric(equity, errors="coerce").to_numpy(dtype=float)
    pr = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    vm = pd.Series(valid_mask).fillna(False).astype(bool).to_numpy(dtype=bool)
    n = int(len(eq))
    h = int(max(1, horizon))
    thr = -abs(float(dd_threshold))
    leads: list[int] = []
    for i in range(n):
        if not vm[i] or pr[i] != 1:
            continue
        base = eq[i]
        if (not np.isfinite(base)) or base <= 0.0:
            continue
        jmax = min(n - 1, i + h)
        found = False
        for j in range(i + 1, jmax + 1):
            future = eq[j]
            if not np.isfinite(future):
                continue
            dd = (future / base) - 1.0
            if dd <= thr:
                leads.append(int(j - i))
                found = True
                break
        if not found:
            continue
    if not leads:
        return float("nan")
    return float(np.median(np.asarray(leads, dtype=float)))


def _median_lead_time_regime(
    *,
    regime: pd.Series,
    pred: pd.Series,
    valid_mask: pd.Series,
    horizon: int,
    target_regimes: set[str],
) -> float:
    rg = regime.astype(str).str.lower().to_numpy(dtype=object)
    pr = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    vm = pd.Series(valid_mask).fillna(False).astype(bool).to_numpy(dtype=bool)
    n = int(len(rg))
    h = int(max(1, horizon))
    leads: list[int] = []
    for i in range(n):
        if not vm[i] or pr[i] != 1:
            continue
        jmax = min(n - 1, i + h)
        for j in range(i + 1, jmax + 1):
            if str(rg[j]).lower() in target_regimes:
                leads.append(int(j - i))
                break
    if not leads:
        return float("nan")
    return float(np.median(np.asarray(leads, dtype=float)))


def _build_universe_rows(
    *,
    universe_name: str,
    data: pd.DataFrame,
    horizons: list[int],
    quantiles: list[float],
    split_cutoffs: list[pd.Timestamp],
    dd_threshold: float,
    target_regimes: set[str],
    use_flags_valid_train: int,
    random_iters: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    if data.empty:
        return rows, split_summary

    d = data.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if d.empty:
        return rows, split_summary

    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["flags_valid"] = pd.to_numeric(d.get("flags_valid"), errors="coerce").fillna(0).astype(int)
    d["equity"] = pd.to_numeric(d["equity"], errors="coerce")
    has_regime = "regime" in d.columns
    if has_regime:
        d["regime"] = d["regime"].astype(str).str.lower()
        d["pred_regime"] = d["regime"].isin(target_regimes).astype("Int64")
    else:
        d["pred_regime"] = pd.Series([pd.NA] * len(d), dtype="Int64")

    for split_idx, cutoff in enumerate(split_cutoffs):
        split_name = f"split_{split_idx + 1}"
        train_mask = d["date"] <= cutoff
        test_mask = d["date"] > cutoff
        train_n = int(train_mask.sum())
        test_n = int(test_mask.sum())
        split_summary[split_name] = {
            "cutoff": str(cutoff.date()),
            "train_rows": train_n,
            "test_rows": test_n,
        }
        if train_n <= 1 or test_n <= 1:
            continue

        if int(use_flags_valid_train) == 1:
            train_for_threshold = train_mask & (d["flags_valid"] == 1)
        else:
            train_for_threshold = train_mask

        for horizon in horizons:
            y_drawdown = build_event_label(
                equity=d["equity"],
                horizon_days=int(horizon),
                dd_threshold=float(dd_threshold),
            )
            y_regime = (
                build_regime_future_event_label(
                    regime=d["regime"],
                    horizon_days=int(horizon),
                    target_regimes=target_regimes,
                )
                if has_regime
                else pd.Series([pd.NA] * len(d), dtype="Int64")
            )

            for q in quantiles:
                thr = threshold_from_train(score=d["score"], train_mask=train_for_threshold, q=float(q))
                pred_score = (pd.to_numeric(d["score"], errors="coerce") >= float(thr)).astype("Int64")
                model_specs: list[tuple[str, pd.Series]] = [("score_only", pred_score)]
                if has_regime:
                    pred_combined = (
                        (pred_score.fillna(0).astype(int) == 1)
                        | (d["pred_regime"].fillna(0).astype(int) == 1)
                    ).astype("Int64")
                    model_specs.extend([("regime_only", d["pred_regime"]), ("combined_or", pred_combined)])

                eval_specs: list[tuple[str, pd.Series]] = [("drawdown", y_drawdown)]
                if has_regime:
                    eval_specs.append(("regime_entry", y_regime))

                for gt_name, y_true_all in eval_specs:
                    valid = test_mask & y_true_all.notna()
                    y_true = y_true_all.loc[valid]
                    for model_name, pred in model_specs:
                        y_pred = pd.Series(pred, index=d.index).loc[valid]
                        rep = classification_report_binary(y_true, y_pred)

                        rb = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
                        if model_name == "score_only":
                            rb = _random_baseline(
                                y_true,
                                alert_rate=_safe_float(rep.get("alert_rate")),
                                n_iter=int(random_iters),
                                seed=int(seed + split_idx * 10_000 + int(horizon) * 100 + int(q * 1000)),
                            )

                        lead_time = float("nan")
                        if gt_name == "drawdown":
                            lead_time = _median_lead_time_drawdown(
                                equity=d["equity"],
                                pred=pred,
                                valid_mask=valid,
                                horizon=int(horizon),
                                dd_threshold=float(dd_threshold),
                            )
                        elif gt_name == "regime_entry" and has_regime:
                            lead_time = _median_lead_time_regime(
                                regime=d["regime"],
                                pred=pred,
                                valid_mask=valid,
                                horizon=int(horizon),
                                target_regimes=target_regimes,
                            )

                        rows.append(
                            {
                                "universe": str(universe_name),
                                "split": split_name,
                                "cutoff": str(cutoff.date()),
                                "horizon_days": int(horizon),
                                "quantile": float(q),
                                "threshold": _safe_float(thr),
                                "ground_truth": str(gt_name),
                                "model": str(model_name),
                                "train_rows": train_n,
                                "test_rows": test_n,
                                "test_rows_labeled": int(valid.sum()),
                                "flags_valid_train_rows": int(train_for_threshold.sum()),
                                "n": _safe_float(rep.get("n")),
                                "event_rate": _safe_float(rep.get("event_rate")),
                                "alert_rate": _safe_float(rep.get("alert_rate")),
                                "precision": _safe_float(rep.get("precision")),
                                "recall": _safe_float(rep.get("recall")),
                                "f1": _safe_float(rep.get("f1")),
                                "accuracy": _safe_float(rep.get("accuracy")),
                                "median_lead_time": _safe_float(lead_time),
                                "random_precision": _safe_float(rb.get("precision")),
                                "random_recall": _safe_float(rb.get("recall")),
                                "random_f1": _safe_float(rb.get("f1")),
                                "lift_precision_vs_random": _safe_lift(_safe_float(rep.get("precision")), _safe_float(rb.get("precision"))),
                                "lift_recall_vs_random": _safe_lift(_safe_float(rep.get("recall")), _safe_float(rb.get("recall"))),
                                "lift_f1_vs_random": _safe_lift(_safe_float(rep.get("f1")), _safe_float(rb.get("f1"))),
                            }
                        )
    return rows, split_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Epistemic diagnostics with split + quantile curves + random baseline lift.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--universe", type=str, default="global", help="global | all | gics:<name> | internal:<name>")
    ap.add_argument("--horizons", type=str, default="5,10")
    ap.add_argument("--quantiles", type=str, default="0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--split-cutoffs", type=str, default="2023-12-31,2024-06-30")
    ap.add_argument("--drawdown-threshold", type=float, default=0.05)
    ap.add_argument("--target-regimes", type=str, default="stress,transition")
    ap.add_argument("--random-iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--use-flags-valid-train", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_lab_run()
    selector = str(args.universe).strip().lower() or "global"
    if str(args.outdir).strip():
        outdir = ROOT / str(args.outdir).strip()
    else:
        prefix = "epistemic_hierarchical" if selector != "global" else "epistemic_diagnostics"
        outdir = ROOT / "results" / f"{prefix}_{_run_id()}"
    outdir.mkdir(parents=True, exist_ok=True)

    horizons = _parse_int_list(str(args.horizons), default=[5, 10])
    quantiles = _parse_float_list(str(args.quantiles), default=[0.75, 0.80, 0.85, 0.90, 0.95])
    split_cutoffs = _parse_date_list(str(args.split_cutoffs), default=["2023-12-31", "2024-06-30"])
    target_regimes = {x.strip().lower() for x in str(args.target_regimes).split(",") if x.strip()}
    if not target_regimes:
        target_regimes = {"stress", "transition"}

    universes = _discover_universes(run_dir=run_dir, selector=selector)
    if not universes:
        raise SystemExit(f"no universes found for selector={selector}")

    global_ref = _load_global_reference(run_dir)
    returns_core = _load_returns_core(run_dir)
    all_rows: list[dict[str, Any]] = []
    split_summary_by_universe: dict[str, Any] = {}
    universe_meta_rows: list[dict[str, Any]] = []

    for u in universes:
        name = str(u["name"])
        merged = _merge_universe_data(universe=u, global_ref=global_ref, returns_core=returns_core)
        if merged.empty:
            universe_meta_rows.append({"universe": name, "rows_merged": 0, "status": "skipped"})
            continue
        rows, split_summary = _build_universe_rows(
            universe_name=name,
            data=merged,
            horizons=horizons,
            quantiles=quantiles,
            split_cutoffs=split_cutoffs,
            dd_threshold=float(args.drawdown_threshold),
            target_regimes=target_regimes,
            use_flags_valid_train=int(args.use_flags_valid_train),
            random_iters=int(args.random_iters),
            seed=int(args.seed),
        )
        all_rows.extend(rows)
        split_summary_by_universe[name] = split_summary
        universe_meta_rows.append(
            {
                "universe": name,
                "rows_merged": int(merged.shape[0]),
                "score_path": str(u.get("score_path", "")),
                "assets_path": str(u.get("assets_path", "")),
                "status": "ok",
            }
        )

    report_df = pd.DataFrame(all_rows)
    if report_df.empty:
        raise SystemExit("empty diagnostics rows (check universes/labels)")

    curve_csv = outdir / "epistemic_curve.csv"
    summary_csv = outdir / "epistemic_summary.csv"
    legacy_curve_csv = outdir / "epistemic_diagnostics_curve.csv"
    legacy_summary_csv = outdir / "epistemic_diagnostics_summary.csv"
    report_df.to_csv(curve_csv, index=False)
    report_df.to_csv(legacy_curve_csv, index=False)

    summary_rows: list[dict[str, Any]] = []
    score_only = report_df[report_df["model"] == "score_only"].copy()
    grouped = score_only.groupby(["universe", "split", "horizon_days", "ground_truth"], dropna=False)
    for (universe, split, horizon, gt_name), grp in grouped:
        g = grp.sort_values("quantile")
        best_idx = g["f1"].astype(float).idxmax() if g["f1"].notna().any() else None
        best = g.loc[best_idx] if best_idx is not None else None
        summary_rows.append(
            {
                "universe": str(universe),
                "split": str(split),
                "horizon_days": int(horizon),
                "ground_truth": str(gt_name),
                "event_rate_min": _safe_float(g["event_rate"].min()),
                "event_rate_max": _safe_float(g["event_rate"].max()),
                "alert_rate_min": _safe_float(g["alert_rate"].min()),
                "alert_rate_max": _safe_float(g["alert_rate"].max()),
                "best_quantile_by_f1": _safe_float(best["quantile"]) if best is not None else float("nan"),
                "best_f1": _safe_float(best["f1"]) if best is not None else float("nan"),
                "best_recall": _safe_float(best["recall"]) if best is not None else float("nan"),
                "best_precision": _safe_float(best["precision"]) if best is not None else float("nan"),
                "best_lift_precision": _safe_float(best["lift_precision_vs_random"]) if best is not None else float("nan"),
                "best_median_lead_time": _safe_float(best["median_lead_time"]) if best is not None else float("nan"),
                "recall_all_zero_curve": bool((g["recall"].fillna(0.0) <= 0.0).all()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_csv(legacy_summary_csv, index=False)

    universe_meta_df = pd.DataFrame(universe_meta_rows)
    universe_meta_path = outdir / "epistemic_universe_map.csv"
    universe_meta_df.to_csv(universe_meta_path, index=False)

    summary_json = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "universe_selector": selector,
        "universes_evaluated": sorted(report_df["universe"].astype(str).unique().tolist()),
        "params": {
            "horizons": horizons,
            "quantiles": quantiles,
            "split_cutoffs": [str(x.date()) for x in split_cutoffs],
            "drawdown_threshold": float(args.drawdown_threshold),
            "target_regimes": sorted(target_regimes),
            "random_iters": int(args.random_iters),
            "seed": int(args.seed),
            "use_flags_valid_train": int(args.use_flags_valid_train),
        },
        "counts": {
            "rows_curve": int(report_df.shape[0]),
            "rows_summary": int(summary_df.shape[0]),
            "universes_requested": int(len(universes)),
            "universes_evaluated": int(report_df["universe"].nunique()),
        },
        "splits_by_universe": split_summary_by_universe,
        "files": {
            "curve_csv": str(curve_csv),
            "summary_csv": str(summary_csv),
            "legacy_curve_csv": str(legacy_curve_csv),
            "legacy_summary_csv": str(legacy_summary_csv),
            "universe_map_csv": str(universe_meta_path),
            "summary_json": str(outdir / "epistemic_diagnostics_summary.json"),
        },
    }
    summary_path = outdir / "epistemic_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/structural/run_epistemic_diagnostics.py",
        params=summary_json["params"] | {"universe_selector": selector},
        paths={
            "curve_csv": str(curve_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_path),
            "universe_map_csv": str(universe_meta_path),
        },
        gates={
            "curve_nonempty": bool(report_df.shape[0] > 0),
            "summary_nonempty": bool(summary_df.shape[0] > 0),
            "summary_written": bool(summary_path.exists()),
        },
        extra={"run_dir": str(run_dir), "universes": summary_json["universes_evaluated"]},
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "universes": summary_json["universes_evaluated"],
                "curve_rows": int(report_df.shape[0]),
                "summary_rows": int(summary_df.shape[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
