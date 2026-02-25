#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _clip_budget(x: float) -> float:
    return float(min(0.50, max(0.01, x)))


def _parse_budget_list(text: str) -> list[float]:
    vals: list[float] = []
    for tok in str(text or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(_clip_budget(float(t)))
        except (TypeError, ValueError):
            continue
    out = sorted(set(vals))
    return out if out else [0.10, 0.15, 0.20]


def _latest_lab_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "hierarchical").exists():
            return d
    raise FileNotFoundError("no lab run found")


def _verdict_tag(*, lift: float, recall: float, months: int) -> str:
    if months < 6:
        return "insuficiente"
    if np.isfinite(lift) and np.isfinite(recall) and (lift >= 1.2) and (recall >= 0.18):
        return "forte"
    if np.isfinite(lift) and np.isfinite(recall) and (lift >= 1.0) and (recall >= 0.08):
        return "moderado"
    return "fraco"


def _build_yearly_performance(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = monthly.copy()
    x["month_start"] = pd.to_datetime(x["month_start"], errors="coerce")
    x = x.dropna(subset=["month_start"]).copy()
    x["year"] = x["month_start"].dt.year.astype(int)
    for c in ["f1", "precision", "recall", "lift_precision_vs_random", "event_rate", "alert_rate"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    grp = (
        x.groupby(["year", "target", "mode", "model"], dropna=False)
        .agg(
            months=("month", "nunique"),
            months_with_f1=("f1", lambda s: int(s.notna().sum())),
            f1_mean=("f1", "mean"),
            f1_mean_nan0=("f1", lambda s: float(s.fillna(0.0).mean())),
            precision_mean_nan0=("precision", lambda s: float(s.fillna(0.0).mean())),
            recall_mean_nan0=("recall", lambda s: float(s.fillna(0.0).mean())),
            lift_mean_nan0=("lift_precision_vs_random", lambda s: float(s.fillna(0.0).mean())),
            event_rate_mean=("event_rate", "mean"),
            alert_rate_mean=("alert_rate", "mean"),
        )
        .reset_index()
    )
    grp = grp.sort_values(["year", "target", "mode", "f1_mean_nan0", "lift_mean_nan0"], ascending=[True, True, True, False, False]).reset_index(drop=True)

    best = (
        grp.sort_values(["year", "target", "mode", "f1_mean_nan0", "lift_mean_nan0"], ascending=[True, True, True, False, False])
        .groupby(["year", "target", "mode"], as_index=False)
        .first()
    )
    best["verdict"] = best.apply(
        lambda r: _verdict_tag(
            lift=_safe_float(r.get("lift_mean_nan0")),
            recall=_safe_float(r.get("recall_mean_nan0")),
            months=int(r.get("months", 0)),
        ),
        axis=1,
    )
    return grp, best


def _build_yearly_sector_asset_rankings(asset_global: pd.DataFrame, *, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = asset_global.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).copy()
    x["year"] = x["date"].dt.year.astype(int)
    x["impact_global"] = pd.to_numeric(x["impact_global"], errors="coerce").fillna(0.0)

    if "sector_gics" not in x.columns:
        x["sector_gics"] = "unknown"
    if "ticker" not in x.columns:
        x["ticker"] = x.get("asset_id", "")
    x["sector_gics"] = x["sector_gics"].astype(str)
    x["ticker"] = x["ticker"].astype(str)
    x["asset_id"] = x["asset_id"].astype(str)

    sec = (
        x.groupby(["year", "sector_gics"], as_index=False)["impact_global"]
        .mean()
        .rename(columns={"sector_gics": "sector", "impact_global": "mean_daily_impact_global"})
        .sort_values(["year", "mean_daily_impact_global"], ascending=[True, False])
        .reset_index(drop=True)
    )
    sec["rank"] = sec.groupby("year")["mean_daily_impact_global"].rank(method="first", ascending=False).astype(int)
    sec_top = sec[sec["rank"] <= int(max(1, top_n))].copy()

    ast = (
        x.groupby(["year", "asset_id", "ticker", "sector_gics"], as_index=False)["impact_global"]
        .mean()
        .rename(columns={"impact_global": "mean_daily_impact_global"})
        .sort_values(["year", "mean_daily_impact_global"], ascending=[True, False])
        .reset_index(drop=True)
    )
    ast["rank"] = ast.groupby("year")["mean_daily_impact_global"].rank(method="first", ascending=False).astype(int)
    ast_top = ast[ast["rank"] <= int(max(1, top_n))].copy()
    return sec_top, ast_top


def _build_next_month_indication(run_dir: Path, impact_dir: Path) -> dict[str, Any]:
    hier = run_dir / "hierarchical"
    score_path = hier / "diagnostics_global_score_daily.csv"
    diag_path = hier / "diagnostics_global_daily.csv"
    bt_path = run_dir / "backtest_regime_T120.csv"
    cross_path = hier / "cross_sector_global_gics_daily.csv"
    asset_global_path = impact_dir / "asset_global_impact_daily.csv"

    out: dict[str, Any] = {
        "status": "ok",
        "as_of_date": None,
        "data_last_date": None,
        "risk_level_next_month": "unknown",
        "rationale": [],
    }

    if (not score_path.exists()) or (not diag_path.exists()) or (not bt_path.exists()) or (not asset_global_path.exists()):
        out["status"] = "missing_inputs"
        return out

    s = pd.read_csv(score_path)
    d = pd.read_csv(diag_path)
    bt = pd.read_csv(bt_path)
    ag = pd.read_csv(asset_global_path)

    for df in (s, d, bt, ag):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    s = s.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    bt = bt.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    ag = ag.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if s.empty or d.empty or bt.empty or ag.empty:
        out["status"] = "empty_inputs"
        return out

    latest_date = min(s["date"].max(), d["date"].max(), bt["date"].max(), ag["date"].max())
    s_tail = s[s["date"] <= latest_date].tail(252).copy()
    d_last = d[d["date"] <= latest_date].tail(1)
    bt_last = bt[bt["date"] <= latest_date].tail(1)
    ag_last = ag[ag["date"] == latest_date].copy()
    if s_tail.empty or d_last.empty or bt_last.empty or ag_last.empty:
        out["status"] = "insufficient_latest_rows"
        return out

    score_last = _safe_float(s_tail["score"].iloc[-1])
    q70 = _safe_float(s_tail["score"].quantile(0.70))
    q90 = _safe_float(s_tail["score"].quantile(0.90))
    regime_last = str(bt_last["regime"].iloc[-1]).lower() if "regime" in bt_last.columns else "unknown"
    exposure_last = _safe_float(bt_last["exposure"].iloc[-1]) if "exposure" in bt_last.columns else float("nan")
    q_last = _safe_float(d_last["Q"].iloc[-1]) if "Q" in d_last.columns else float("nan")
    n_used_last = _safe_float(d_last["N_used"].iloc[-1]) if "N_used" in d_last.columns else float("nan")

    if np.isfinite(score_last) and np.isfinite(q90) and score_last >= q90:
        risk = "alto"
    elif regime_last in {"stress", "transition"}:
        risk = "alto"
    elif np.isfinite(score_last) and np.isfinite(q70) and score_last >= q70:
        risk = "moderado"
    else:
        risk = "baixo"

    sec_rows: list[dict[str, Any]] = []
    ag_last["impact_global"] = pd.to_numeric(ag_last["impact_global"], errors="coerce").fillna(0.0)
    if "sector_gics" not in ag_last.columns:
        ag_last["sector_gics"] = "unknown"
    sec = (
        ag_last.groupby("sector_gics", as_index=False)["impact_global"]
        .sum()
        .rename(columns={"sector_gics": "sector", "impact_global": "impact"})
        .sort_values("impact", ascending=False)
        .head(5)
    )
    sec_rows = sec.to_dict(orient="records")

    overlap_rows: list[dict[str, Any]] = []
    if cross_path.exists():
        cr = pd.read_csv(cross_path)
        cr["date"] = pd.to_datetime(cr["date"], errors="coerce")
        cr = cr.dropna(subset=["date"]).sort_values("date")
        cr = cr[cr["date"] == latest_date].copy()
        if (not cr.empty) and ("overlap_sector_global" in cr.columns):
            cr["overlap_sector_global"] = pd.to_numeric(cr["overlap_sector_global"], errors="coerce")
            overlap_rows = (
                cr.sort_values("overlap_sector_global", ascending=False)[["sector", "overlap_sector_global"]]
                .head(5)
                .to_dict(orient="records")
            )

    out.update(
        {
            "as_of_date": str(latest_date.date()),
            "data_last_date": str(latest_date.date()),
            "risk_level_next_month": str(risk),
            "global_state": {
                "score_last": score_last,
                "score_q70_252d": q70,
                "score_q90_252d": q90,
                "regime_last": regime_last,
                "exposure_last": exposure_last,
                "Q_last": q_last,
                "N_used_last": n_used_last,
            },
            "top_sectors_by_impact": sec_rows,
            "top_sectors_by_overlap": overlap_rows,
            "rationale": [
                "indicacao baseada em score global rolling (252d), regime atual e concentracao setorial",
                "classificacao: baixo/moderado/alto para o proximo mes operacional",
            ],
        }
    )
    return out


def _load_sector_index(hier_dir: Path) -> dict[tuple[str, str], str]:
    idx_path = hier_dir / "universes" / "sector_universe_index.csv"
    out: dict[tuple[str, str], str] = {}
    if not idx_path.exists():
        return out
    d = pd.read_csv(idx_path)
    if d.empty:
        return out
    for _, r in d.iterrows():
        kind = str(r.get("kind", "")).strip().lower()
        slug = str(r.get("slug", "")).strip()
        sec = str(r.get("sector", "")).strip()
        if kind and slug and sec:
            out[(kind, slug)] = sec
    return out


def _dedupe_alerts(dates: pd.Series, alerts: pd.Series, *, dedupe_days: int) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    a = pd.Series(alerts).fillna(False).astype(bool)
    out = pd.Series(False, index=a.index)
    last_alert: pd.Timestamp | None = None
    gap = int(max(0, dedupe_days))
    for idx in a.index:
        if not bool(a.loc[idx]):
            continue
        dt = d.loc[idx]
        if pd.isna(dt):
            continue
        if (last_alert is None) or ((pd.Timestamp(dt) - pd.Timestamp(last_alert)).days >= gap):
            out.loc[idx] = True
            last_alert = pd.Timestamp(dt)
    return out


def _alert_frame(
    score_df: pd.DataFrame,
    *,
    lookback_days: int = 252,
    alert_budget: float = 0.15,
    dedupe_days: int = 20,
) -> pd.DataFrame:
    x = score_df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["score"] = pd.to_numeric(x["score"], errors="coerce")
    x = x.dropna(subset=["date", "score"]).sort_values("date").reset_index(drop=True)
    if x.empty:
        return pd.DataFrame(columns=["date", "score", "threshold", "raw_alert", "alert"])
    q = float(1.0 - _clip_budget(alert_budget))
    w = int(max(20, lookback_days))
    x["threshold"] = x["score"].rolling(w, min_periods=w).quantile(q).shift(1)
    x["raw_alert"] = (x["score"] >= x["threshold"]) & x["threshold"].notna()
    x["alert"] = _dedupe_alerts(x["date"], x["raw_alert"], dedupe_days=int(max(0, dedupe_days)))
    return x[["date", "score", "threshold", "raw_alert", "alert"]].copy()


def _build_lead_signals(
    run_dir: Path,
    *,
    lead_window_days: int = 30,
    min_event_gap_days: int = 20,
    target_regimes: set[str] | None = None,
    alert_budget: float = 0.15,
    alert_dedupe_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hier = run_dir / "hierarchical"
    bt_path = run_dir / "backtest_regime_T120.csv"
    global_path = hier / "diagnostics_global_score_daily.csv"
    if (not bt_path.exists()) or (not global_path.exists()):
        return (
            pd.DataFrame(columns=["source", "event_date", "alerted", "lead_days", "alert_date"]),
            pd.DataFrame(columns=["source", "events_total", "alerted_events", "alert_rate", "median_lead_days"]),
        )

    bt = pd.read_csv(bt_path)
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
    bt["regime"] = bt.get("regime", "").astype(str).str.lower()
    bt = bt.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return (
            pd.DataFrame(columns=["source", "event_date", "alerted", "lead_days", "alert_date"]),
            pd.DataFrame(columns=["source", "events_total", "alerted_events", "alert_rate", "median_lead_days"]),
        )
    regimes = {str(x).strip().lower() for x in (target_regimes or {"stress", "transition"}) if str(x).strip()}
    if not regimes:
        regimes = {"stress", "transition"}
    event_mask = bt["regime"].isin(regimes)
    prev_event = event_mask.shift(1).fillna(False)
    entries = bt.loc[event_mask & (~prev_event), ["date"]].copy()
    entries["event_date"] = entries["date"].dt.date.astype(str)
    entry_dates_raw = pd.to_datetime(entries["event_date"], errors="coerce").dropna().sort_values().tolist()
    entry_dates: list[pd.Timestamp] = []
    gap_days = int(max(1, min_event_gap_days))
    for ed in entry_dates_raw:
        if (not entry_dates) or ((pd.Timestamp(ed) - pd.Timestamp(entry_dates[-1])).days >= gap_days):
            entry_dates.append(pd.Timestamp(ed))
    if not entry_dates:
        return (
            pd.DataFrame(columns=["source", "event_date", "alerted", "lead_days", "alert_date"]),
            pd.DataFrame(columns=["source", "events_total", "alerted_events", "alert_rate", "median_lead_days"]),
        )

    sources: dict[str, pd.DataFrame] = {}
    g = pd.read_csv(global_path)
    sources["global"] = _alert_frame(
        g,
        lookback_days=252,
        alert_budget=float(alert_budget),
        dedupe_days=int(max(0, alert_dedupe_days)),
    )

    idx = _load_sector_index(hier)
    for kind in ("gics", "internal"):
        pat = f"diagnostics_sector_{kind}_*_score_daily.csv"
        prefix = f"diagnostics_sector_{kind}_"
        for p in sorted(hier.glob(pat)):
            slug = p.stem.replace(prefix, "")
            if slug.endswith("_score_daily"):
                slug = slug[: -len("_score_daily")]
            sec_name = idx.get((kind, slug), slug)
            d = pd.read_csv(p)
            sources[f"{kind}:{sec_name}"] = _alert_frame(
                d,
                lookback_days=252,
                alert_budget=float(alert_budget),
                dedupe_days=int(max(0, alert_dedupe_days)),
            )

    rows: list[dict[str, Any]] = []
    for src, af in sources.items():
        if af.empty:
            continue
        adates = af.loc[af["alert"], "date"].sort_values().to_list()
        raw_rate = _safe_float(pd.to_numeric(af.get("raw_alert"), errors="coerce").fillna(0.0).mean())
        deduped_rate = _safe_float(pd.to_numeric(af.get("alert"), errors="coerce").fillna(0.0).mean())
        for ed in entry_dates:
            lo = pd.Timestamp(ed) - pd.Timedelta(days=int(max(1, lead_window_days)))
            candidates = [d for d in adates if (d >= lo) and (d <= pd.Timestamp(ed))]
            if candidates:
                ad = max(candidates)
                lead = int((pd.Timestamp(ed) - pd.Timestamp(ad)).days)
                rows.append(
                    {
                        "source": str(src),
                        "event_date": str(pd.Timestamp(ed).date()),
                        "alerted": 1,
                        "lead_days": int(lead),
                        "alert_date": str(pd.Timestamp(ad).date()),
                        "daily_alert_rate_raw": raw_rate,
                        "daily_alert_rate_deduped": deduped_rate,
                    }
                )
            else:
                rows.append(
                    {
                        "source": str(src),
                        "event_date": str(pd.Timestamp(ed).date()),
                        "alerted": 0,
                        "lead_days": float("nan"),
                        "alert_date": "",
                        "daily_alert_rate_raw": raw_rate,
                        "daily_alert_rate_deduped": deduped_rate,
                    }
                )
    if not rows:
        return (
            pd.DataFrame(columns=["source", "event_date", "alerted", "lead_days", "alert_date"]),
            pd.DataFrame(columns=["source", "events_total", "alerted_events", "alert_rate", "median_lead_days"]),
        )
    detail = pd.DataFrame(rows).sort_values(["source", "event_date"]).reset_index(drop=True)
    summary = (
        detail.groupby("source", as_index=False)
        .agg(
            events_total=("event_date", "count"),
            alerted_events=("alerted", "sum"),
            median_lead_days=("lead_days", "median"),
            alerted_before_1d=("lead_days", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(-1).ge(1).sum())),
            alerted_before_5d=("lead_days", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(-1).ge(5).sum())),
            daily_alert_rate_raw=("daily_alert_rate_raw", "median"),
            daily_alert_rate_deduped=("daily_alert_rate_deduped", "median"),
        )
        .sort_values(["alerted_events", "median_lead_days"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["alert_rate"] = np.where(summary["events_total"] > 0, summary["alerted_events"] / summary["events_total"], np.nan)
    summary["alert_before_1d_rate"] = np.where(summary["events_total"] > 0, summary["alerted_before_1d"] / summary["events_total"], np.nan)
    summary["alert_before_5d_rate"] = np.where(summary["events_total"] > 0, summary["alerted_before_5d"] / summary["events_total"], np.nan)
    summary["event_set"] = ",".join(sorted(regimes))
    summary["alert_budget"] = float(_clip_budget(alert_budget))
    summary["alert_dedupe_days"] = int(max(0, alert_dedupe_days))
    return detail, summary


def _build_lead_budget_sweep(
    run_dir: Path,
    *,
    budgets: list[float],
    lead_window_days: int,
    min_event_gap_days: int,
    alert_dedupe_days: int,
    target_regimes: set[str] | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for b in sorted(set([_clip_budget(float(x)) for x in budgets])):
        _, summ = _build_lead_signals(
            run_dir=run_dir,
            lead_window_days=int(max(1, lead_window_days)),
            min_event_gap_days=int(max(1, min_event_gap_days)),
            target_regimes=target_regimes,
            alert_budget=float(b),
            alert_dedupe_days=int(max(0, alert_dedupe_days)),
        )
        if summ.empty:
            continue
        rows.append(summ)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["source", "alert_budget"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build year-by-year structural assessment from impact learning outputs.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--impact-dir", type=str, default="")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--alert-budget", type=float, default=0.15)
    ap.add_argument("--alert-budget-sweep", type=str, default="0.10,0.15,0.20")
    ap.add_argument("--alert-dedupe-days", type=int, default=20)
    ap.add_argument("--lead-window-days", type=int, default=30)
    ap.add_argument("--min-event-gap-days", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_lab_run()
    impact_dir = Path(args.impact_dir).resolve() if str(args.impact_dir).strip() else (run_dir / "hierarchical" / "impact_learning")
    if not impact_dir.exists():
        raise SystemExit(f"impact dir not found: {impact_dir}")

    monthly_path = impact_dir / "impact_walkforward_monthly.csv"
    compare_path = impact_dir / "impact_walkforward_monthly_compare.csv"
    asset_global_path = impact_dir / "asset_global_impact_daily.csv"
    if (not monthly_path.exists()) or (not compare_path.exists()) or (not asset_global_path.exists()):
        raise SystemExit("missing required impact outputs (monthly/compare/asset_global)")

    monthly = pd.read_csv(monthly_path)
    compare = pd.read_csv(compare_path)
    asset_global = pd.read_csv(asset_global_path)

    selected_budget = _clip_budget(float(args.alert_budget))
    budget_sweep = _parse_budget_list(str(args.alert_budget_sweep))

    yearly_perf, yearly_best = _build_yearly_performance(monthly=monthly)
    sectors_yearly, assets_yearly = _build_yearly_sector_asset_rankings(asset_global=asset_global, top_n=int(max(1, args.top_n)))
    next_month = _build_next_month_indication(run_dir=run_dir, impact_dir=impact_dir)
    lead_detail, lead_summary = _build_lead_signals(
        run_dir=run_dir,
        lead_window_days=int(max(1, args.lead_window_days)),
        min_event_gap_days=int(max(1, args.min_event_gap_days)),
        target_regimes={"stress", "transition"},
        alert_budget=float(selected_budget),
        alert_dedupe_days=int(max(0, args.alert_dedupe_days)),
    )
    stress_detail, stress_summary = _build_lead_signals(
        run_dir=run_dir,
        lead_window_days=int(max(1, args.lead_window_days)),
        min_event_gap_days=int(max(1, args.min_event_gap_days)),
        target_regimes={"stress"},
        alert_budget=float(selected_budget),
        alert_dedupe_days=int(max(0, args.alert_dedupe_days)),
    )
    sweep_summary = _build_lead_budget_sweep(
        run_dir=run_dir,
        budgets=budget_sweep,
        lead_window_days=int(max(1, args.lead_window_days)),
        min_event_gap_days=int(max(1, args.min_event_gap_days)),
        alert_dedupe_days=int(max(0, args.alert_dedupe_days)),
        target_regimes={"stress"},
    )

    out_prefix = impact_dir / "historical_structure_assessment"
    perf_csv = out_prefix.with_name("historical_structure_yearly_performance.csv")
    best_csv = out_prefix.with_name("historical_structure_yearly_best_models.csv")
    sectors_csv = out_prefix.with_name("historical_structure_yearly_sector_rankings.csv")
    assets_csv = out_prefix.with_name("historical_structure_yearly_asset_rankings.csv")
    next_json = out_prefix.with_name("historical_structure_next_month_indication.json")
    lead_detail_csv = out_prefix.with_name("historical_structure_lead_signals_detail.csv")
    lead_summary_csv = out_prefix.with_name("historical_structure_lead_signals_summary.csv")
    stress_detail_csv = out_prefix.with_name("historical_structure_stress_prealert_detail.csv")
    stress_summary_csv = out_prefix.with_name("historical_structure_stress_prealert_summary.csv")
    budget_sweep_csv = out_prefix.with_name("historical_structure_stress_prealert_budget_sweep.csv")
    summary_json = out_prefix.with_name("historical_structure_summary.json")

    yearly_perf.to_csv(perf_csv, index=False)
    yearly_best.to_csv(best_csv, index=False)
    sectors_yearly.to_csv(sectors_csv, index=False)
    assets_yearly.to_csv(assets_csv, index=False)
    next_json.write_text(json.dumps(next_month, indent=2, ensure_ascii=False), encoding="utf-8")
    lead_detail.to_csv(lead_detail_csv, index=False)
    lead_summary.to_csv(lead_summary_csv, index=False)
    stress_detail.to_csv(stress_detail_csv, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)
    sweep_summary.to_csv(budget_sweep_csv, index=False)

    years = sorted(yearly_best["year"].dropna().astype(int).unique().tolist()) if not yearly_best.empty else []
    exp_best = yearly_best[yearly_best["mode"] == "expanding"].copy() if "mode" in yearly_best.columns else pd.DataFrame()
    verdict_counts = {}
    if not exp_best.empty and "verdict" in exp_best.columns:
        verdict_counts = exp_best["verdict"].value_counts(dropna=False).to_dict()

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "impact_dir": str(impact_dir),
        "data_last_date": str(next_month.get("data_last_date") or next_month.get("as_of_date") or ""),
        "years_covered": years,
        "year_count": int(len(years)),
        "mode_note": "fixed requer historico anterior suficiente; com train_end 2015 e dados iniciando 2016, expanding e o modo valido",
        "lead_event_definition": "entradas em stress/transition com separacao minima configuravel; alerta buscado em janela causal anterior",
        "lead_alert_config": {
            "selected_alert_budget": float(selected_budget),
            "budget_sweep": budget_sweep,
            "alert_dedupe_days": int(max(0, args.alert_dedupe_days)),
            "lead_window_days": int(max(1, args.lead_window_days)),
            "min_event_gap_days": int(max(1, args.min_event_gap_days)),
        },
        "verdict_counts_expanding": verdict_counts,
        "next_month_indication": next_month,
        "lead_signal_summary_top10": lead_summary.head(10).to_dict(orient="records"),
        "stress_prealert_summary_top10": stress_summary.head(10).to_dict(orient="records"),
        "files": {
            "yearly_performance_csv": str(perf_csv),
            "yearly_best_models_csv": str(best_csv),
            "yearly_sector_rankings_csv": str(sectors_csv),
            "yearly_asset_rankings_csv": str(assets_csv),
            "next_month_indication_json": str(next_json),
            "lead_signals_detail_csv": str(lead_detail_csv),
            "lead_signals_summary_csv": str(lead_summary_csv),
            "stress_prealert_detail_csv": str(stress_detail_csv),
            "stress_prealert_summary_csv": str(stress_summary_csv),
            "stress_prealert_budget_sweep_csv": str(budget_sweep_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"status": "ok", "summary_json": str(summary_json), "year_count": int(len(years))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
