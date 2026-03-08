#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import (  # noqa: E402
    apply_net_assumptions,
    blend_profiles,
    load_net_assumption_profiles,
    summarize_net_series,
)
from execution.returns import daily_simple_to_monthly, load_return_series_csv  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _infer_asset_jurisdiction(ticker: str, group: str) -> str:
    tt = str(ticker).strip().upper()
    gg = str(group).strip().lower()
    if tt.endswith(".SA") or gg == "equities_br_bluechips":
        return "br_local"
    return "foreign"


def _load_asset_table(asset_groups_csv: Path, asset_metadata_csv: Path) -> pd.DataFrame:
    groups = pd.read_csv(asset_groups_csv).rename(columns={"asset": "asset_id", "group": "asset_group"})
    meta = pd.read_csv(asset_metadata_csv)
    groups["asset_id"] = groups["asset_id"].astype(str).str.strip()
    meta["asset_id"] = meta["asset_id"].astype(str).str.strip()
    merged = groups.merge(meta, on="asset_id", how="left")
    merged["ticker"] = merged.get("ticker", merged["asset_id"]).astype(str).str.strip()
    merged["asset_group"] = merged["asset_group"].astype(str).str.strip()
    merged["liquidity_proxy"] = pd.to_numeric(merged.get("liquidity_proxy"), errors="coerce").fillna(0.0)
    merged["jurisdiction"] = [
        _infer_asset_jurisdiction(ticker=ticker, group=group)
        for ticker, group in zip(merged["ticker"], merged["asset_group"])
    ]
    return merged[["asset_id", "ticker", "asset_group", "liquidity_proxy", "jurisdiction"]].copy()


def _load_monthly_asset_returns(prices_dir: Path, assets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_frames: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for row in assets.itertuples(index=False):
        ticker = str(row.ticker)
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        try:
            daily = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
        except ValueError:
            continue
        daily = pd.to_numeric(daily, errors="coerce").dropna().astype(float)
        if daily.empty:
            continue
        monthly = daily_simple_to_monthly(daily).rename(str(row.asset_id))
        if monthly.dropna().empty:
            continue
        monthly_frames.append(monthly)
        rows.append(
            {
                "asset_id": str(row.asset_id),
                "ticker": ticker,
                "asset_group": str(row.asset_group),
                "jurisdiction": str(row.jurisdiction),
                "liquidity_proxy": float(row.liquidity_proxy),
                "months_available": int(monthly.dropna().shape[0]),
                "start_month": str(monthly.dropna().index.min()),
                "end_month": str(monthly.dropna().index.max()),
                "gross_total_return": float(np.prod(1.0 + monthly.dropna().to_numpy(dtype=float)) - 1.0),
            }
        )
    if not monthly_frames:
        return pd.DataFrame(), pd.DataFrame(columns=["asset_id", "ticker", "asset_group", "jurisdiction", "liquidity_proxy"])
    monthly_matrix = pd.concat(monthly_frames, axis=1).sort_index()
    return monthly_matrix, pd.DataFrame(rows)


def _build_group_sleeves(
    *,
    asset_table: pd.DataFrame,
    asset_monthly: pd.DataFrame,
    top_assets_per_group: int,
    min_group_assets: int,
    min_assets_present: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    series_map: dict[str, pd.Series] = {}
    for group, sub in asset_table.groupby("asset_group", sort=True):
        chosen = (
            sub.sort_values(["liquidity_proxy", "asset_id"], ascending=[False, True])
            .drop_duplicates(subset=["asset_id"], keep="first")
            .head(int(max(1, top_assets_per_group)))
            .copy()
        )
        available = [asset for asset in chosen["asset_id"].astype(str).tolist() if asset in asset_monthly.columns]
        if len(available) < int(max(1, min_group_assets)):
            continue
        block = asset_monthly[available].apply(pd.to_numeric, errors="coerce")
        present = block.notna().sum(axis=1)
        sleeve = block.mean(axis=1, skipna=True)
        sleeve[present < int(max(1, min_assets_present))] = np.nan
        sleeve = pd.to_numeric(sleeve, errors="coerce")
        if sleeve.dropna().empty:
            continue
        series_map[str(group)] = sleeve.astype(float)
        foreign_share = float((chosen["jurisdiction"].astype(str) == "foreign").mean())
        pool_rows.extend(chosen.assign(months_available=chosen["asset_id"].map(block.count().to_dict()).fillna(0)).to_dict(orient="records"))
        group_rows.append(
            {
                "asset_group": str(group),
                "n_assets_selected": int(len(available)),
                "foreign_share_assets": foreign_share,
                "br_share_assets": float(1.0 - foreign_share),
                "avg_liquidity_proxy": float(pd.to_numeric(chosen["liquidity_proxy"], errors="coerce").mean()),
                "start_month": str(sleeve.dropna().index.min()),
                "end_month": str(sleeve.dropna().index.max()),
                "months_available": int(sleeve.dropna().shape[0]),
            }
        )
    group_monthly = pd.concat(series_map, axis=1).sort_index() if series_map else pd.DataFrame()
    group_meta = pd.DataFrame(group_rows).sort_values("asset_group").reset_index(drop=True)
    asset_pool = pd.DataFrame(pool_rows).sort_values(["asset_group", "liquidity_proxy", "asset_id"], ascending=[True, False, True]).reset_index(drop=True)
    return group_monthly, group_meta, asset_pool


def _load_benchmark_monthly(prices_dir: Path, ticker: str) -> pd.Series:
    path = prices_dir / f"{ticker}.csv"
    daily = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
    return daily_simple_to_monthly(daily).rename(ticker)


def _calc_total_return(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)


def _build_static_combo(
    group_monthly: pd.DataFrame,
    group_meta: pd.DataFrame,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    block = group_monthly[list(groups)].apply(pd.to_numeric, errors="coerce")
    monthly = block.mean(axis=1, skipna=True)
    monthly = monthly[block.notna().sum(axis=1) > 0].astype(float)
    meta_map = group_meta.set_index("asset_group")
    foreign_share = float(pd.to_numeric(meta_map.reindex(list(groups))["foreign_share_assets"], errors="coerce").fillna(0.0).mean())
    turnover = pd.Series(np.zeros(len(monthly), dtype=float), index=monthly.index, dtype=float)
    return {
        "candidate_id": "static_" + "__".join(groups),
        "kind": "group_static_combo",
        "groups": list(groups),
        "gross_monthly": monthly,
        "turnover_monthly": turnover,
        "foreign_share_monthly": pd.Series(np.full(len(monthly), foreign_share, dtype=float), index=monthly.index, dtype=float),
    }


def _l1_turnover(prev_w: dict[str, float], next_w: dict[str, float]) -> float:
    keys = sorted(set(prev_w) | set(next_w))
    return 0.5 * float(sum(abs(float(prev_w.get(k, 0.0)) - float(next_w.get(k, 0.0))) for k in keys))


def _build_dynamic_momentum(
    group_monthly: pd.DataFrame,
    group_meta: pd.DataFrame,
    *,
    lookback_months: int,
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prev_weights: dict[str, float] = {}
    meta_map = group_meta.set_index("asset_group")
    months = list(group_monthly.index.astype(str))
    for pos, ym in enumerate(months):
        if pos < int(max(1, lookback_months)):
            continue
        hist = group_monthly.iloc[pos - int(lookback_months) : pos].apply(pd.to_numeric, errors="coerce")
        scores: dict[str, float] = {}
        for group in hist.columns:
            s = pd.to_numeric(hist[group], errors="coerce").dropna().astype(float)
            if s.shape[0] < int(max(2, lookback_months // 2)):
                continue
            scores[str(group)] = _calc_total_return(s)
        if not scores:
            continue
        selected = [g for g, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: int(max(1, top_k))]]
        realized = pd.to_numeric(group_monthly.loc[ym, selected], errors="coerce").dropna().astype(float)
        if realized.empty:
            continue
        weights = {g: 1.0 / float(len(selected)) for g in selected}
        foreign_share = float(pd.to_numeric(meta_map.reindex(selected)["foreign_share_assets"], errors="coerce").fillna(0.0).mean())
        rows.append(
            {
                "ym": str(ym),
                "ret": float(realized.mean()),
                "turnover": _l1_turnover(prev_weights, weights),
                "foreign_share": foreign_share,
                "selected_groups": ",".join(selected),
            }
        )
        prev_weights = weights
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return {
            "candidate_id": f"dynamic_lb{lookback_months}_k{top_k}",
            "kind": "group_dynamic_momentum",
            "groups": [],
            "gross_monthly": pd.Series(dtype=float),
            "turnover_monthly": pd.Series(dtype=float),
            "foreign_share_monthly": pd.Series(dtype=float),
        }
    monthly["ym"] = monthly["ym"].astype(str)
    return {
        "candidate_id": f"dynamic_lb{lookback_months}_k{top_k}",
        "kind": "group_dynamic_momentum",
        "groups": [],
        "gross_monthly": pd.Series(monthly["ret"].to_numpy(dtype=float), index=monthly["ym"], dtype=float),
        "turnover_monthly": pd.Series(monthly["turnover"].to_numpy(dtype=float), index=monthly["ym"], dtype=float),
        "foreign_share_monthly": pd.Series(monthly["foreign_share"].to_numpy(dtype=float), index=monthly["ym"], dtype=float),
        "selected_groups_csv": monthly[["ym", "selected_groups"]].copy(),
    }


def _candidate_status(edge_vs_spy_net: float, sharpe_net: float, months: int) -> str:
    if months < 36:
        return "watch"
    if np.isfinite(edge_vs_spy_net) and edge_vs_spy_net > 0.0 and np.isfinite(sharpe_net) and sharpe_net >= 0.5:
        return "keep"
    if np.isfinite(edge_vs_spy_net) and edge_vs_spy_net > -0.10:
        return "watch"
    return "kill"


def _evaluate_candidate(
    *,
    candidate: dict[str, Any],
    benchmark_monthly: dict[str, pd.Series],
    net_profiles: dict[str, Any],
) -> dict[str, Any]:
    gross = pd.to_numeric(candidate["gross_monthly"], errors="coerce").dropna().astype(float)
    if gross.empty:
        return {}
    turnover = pd.to_numeric(candidate["turnover_monthly"], errors="coerce").reindex(gross.index).fillna(0.0).astype(float)
    foreign_share_monthly = pd.to_numeric(candidate["foreign_share_monthly"], errors="coerce").reindex(gross.index).fillna(0.0).astype(float)
    avg_foreign_share = float(foreign_share_monthly.mean()) if not foreign_share_monthly.empty else 0.0

    foreign_profile = net_profiles["profiles"]["foreign_financial_brazil_resident"]
    br_profile = net_profiles["profiles"]["br_local_equity"]
    blended_profile = blend_profiles(avg_foreign_share, foreign_profile=foreign_profile, br_profile=br_profile)

    net_foreign = apply_net_assumptions(gross, turnover, profile=foreign_profile, periods_index=gross.index)
    net_br = apply_net_assumptions(gross, turnover, profile=br_profile, periods_index=gross.index)
    net_blended = apply_net_assumptions(gross, turnover, profile=blended_profile, periods_index=gross.index)

    spy = pd.to_numeric(benchmark_monthly["SPY"], errors="coerce").reindex(gross.index).dropna().astype(float)
    spy_aligned = gross.reindex(spy.index).dropna().astype(float)
    spy = spy.reindex(spy_aligned.index).astype(float)
    spy_net = apply_net_assumptions(
        spy,
        pd.Series(np.zeros(len(spy), dtype=float), index=spy.index, dtype=float),
        profile=foreign_profile,
        periods_index=spy.index,
    )

    sixty = pd.to_numeric(benchmark_monthly["sixty_forty"], errors="coerce").reindex(gross.index).dropna().astype(float)
    eqw = pd.to_numeric(benchmark_monthly["group_eqw"], errors="coerce").reindex(gross.index).dropna().astype(float)

    gross_summary = summarize_return_series(gross, periods_per_year=12)
    net_blended_summary = summarize_net_series(net_blended, periods_per_year=12)
    net_foreign_summary = summarize_net_series(net_foreign, periods_per_year=12)
    net_br_summary = summarize_net_series(net_br, periods_per_year=12)
    spy_summary = summarize_return_series(spy, periods_per_year=12)
    spy_net_summary = summarize_net_series(spy_net, periods_per_year=12)
    sixty_summary = summarize_return_series(sixty, periods_per_year=12)
    eqw_summary = summarize_return_series(eqw, periods_per_year=12)

    edge_spy_gross = _safe_float(gross_summary.get("total_return")) - _safe_float(spy_summary.get("total_return"))
    edge_spy_net = _safe_float(net_blended_summary["net"].get("total_return")) - _safe_float(spy_net_summary["net"].get("total_return"))
    status = _candidate_status(
        edge_vs_spy_net=edge_spy_net,
        sharpe_net=_safe_float(net_blended_summary["net"].get("sharpe")),
        months=int(gross.shape[0]),
    )

    return {
        "candidate_id": str(candidate["candidate_id"]),
        "kind": str(candidate["kind"]),
        "groups": ",".join(candidate.get("groups", [])),
        "n_groups": int(len(candidate.get("groups", []))),
        "months": int(gross.shape[0]),
        "avg_turnover_monthly": float(turnover.mean()) if not turnover.empty else 0.0,
        "avg_foreign_share": avg_foreign_share,
        "avg_br_share": float(1.0 - avg_foreign_share),
        "gross_total_return": _safe_float(gross_summary.get("total_return")),
        "gross_ann_return": _safe_float(gross_summary.get("annualized_return")),
        "gross_sharpe": _safe_float(gross_summary.get("sharpe")),
        "gross_max_drawdown": _safe_float(gross_summary.get("max_drawdown")),
        "net_blended_total_return": _safe_float(net_blended_summary["net"].get("total_return")),
        "net_blended_ann_return": _safe_float(net_blended_summary["net"].get("annualized_return")),
        "net_blended_sharpe": _safe_float(net_blended_summary["net"].get("sharpe")),
        "net_blended_max_drawdown": _safe_float(net_blended_summary["net"].get("max_drawdown")),
        "net_foreign_ann_return": _safe_float(net_foreign_summary["net"].get("annualized_return")),
        "net_br_ann_return": _safe_float(net_br_summary["net"].get("annualized_return")),
        "spy_ann_return": _safe_float(spy_summary.get("annualized_return")),
        "spy_net_ann_return": _safe_float(spy_net_summary["net"].get("annualized_return")),
        "sixty_forty_ann_return": _safe_float(sixty_summary.get("annualized_return")),
        "group_eqw_ann_return": _safe_float(eqw_summary.get("annualized_return")),
        "edge_vs_spy_gross_total_return": edge_spy_gross,
        "edge_vs_spy_net_total_return": edge_spy_net,
        "status": status,
        "registry_reason": "keep" if status == "keep" else ("watch" if status == "watch" else "kill"),
        "selected_groups_path": "",
    }


def _build_research_rows(
    results_df: pd.DataFrame,
    *,
    outdir: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results_df.to_dict(orient="records"):
        rows.append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": str(row["candidate_id"]),
                "label": str(row["candidate_id"]),
                "methodology": str(row["kind"]),
                "status": str(row["status"]),
                "gross_ann_return": _safe_float(row.get("gross_ann_return")),
                "net_ann_return": _safe_float(row.get("net_blended_ann_return")),
                "gross_total_return": _safe_float(row.get("gross_total_return")),
                "net_total_return": _safe_float(row.get("net_blended_total_return")),
                "sharpe": _safe_float(row.get("net_blended_sharpe")),
                "max_drawdown": _safe_float(row.get("net_blended_max_drawdown")),
                "edge_vs_spy_net_total_return": _safe_float(row.get("edge_vs_spy_net_total_return")),
                "avg_foreign_share": _safe_float(row.get("avg_foreign_share")),
                "groups": str(row.get("groups", "")),
                "artifacts": {
                    "suite_dir": str(outdir),
                    "summary_json": str(summary_path),
                },
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Pesquisa de metodologias de lucro por grupos de ativos.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--combo-size-max", type=int, default=3)
    ap.add_argument("--top-assets-per-group", type=int, default=8)
    ap.add_argument("--min-group-assets", type=int, default=4)
    ap.add_argument("--min-assets-present", type=int, default=3)
    ap.add_argument("--dynamic-lookbacks", default="3,6,12")
    ap.add_argument("--dynamic-topk", default="2,3,4")
    ap.add_argument("--outdir-root", default="results/validation/profit_group_methodology_suite")
    args = ap.parse_args()

    prices_dir = (ROOT / args.prices_dir).resolve()
    asset_groups_csv = (ROOT / args.asset_groups).resolve()
    asset_metadata_csv = (ROOT / args.asset_metadata).resolve()
    net_cfg_path = (ROOT / args.net_assumptions_config).resolve()
    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    asset_table = _load_asset_table(asset_groups_csv, asset_metadata_csv)
    asset_monthly, asset_viability = _load_monthly_asset_returns(prices_dir, asset_table)
    if asset_monthly.empty:
        raise SystemExit("no asset monthly returns loaded")

    group_monthly, group_meta, group_asset_pool = _build_group_sleeves(
        asset_table=asset_table,
        asset_monthly=asset_monthly,
        top_assets_per_group=int(args.top_assets_per_group),
        min_group_assets=int(args.min_group_assets),
        min_assets_present=int(args.min_assets_present),
    )
    if group_monthly.empty:
        raise SystemExit("no viable group sleeves built")

    spy = _load_benchmark_monthly(prices_dir, "SPY")
    bond = _load_benchmark_monthly(prices_dir, "IEF" if (prices_dir / "IEF.csv").exists() else "SHY")
    common_index = sorted(set(group_monthly.index.astype(str)) & set(spy.index.astype(str)) & set(bond.index.astype(str)))
    group_monthly = group_monthly.reindex(common_index)
    spy = spy.reindex(common_index)
    bond = bond.reindex(common_index)
    benchmark_monthly = {
        "SPY": spy.astype(float),
        "sixty_forty": (0.6 * spy.astype(float) + 0.4 * bond.astype(float)).astype(float),
        "group_eqw": group_monthly.mean(axis=1, skipna=True).astype(float),
    }
    benchmark_monthly_df = pd.DataFrame(benchmark_monthly)

    static_candidates: list[dict[str, Any]] = []
    groups = sorted(group_monthly.columns.astype(str).tolist())
    for size in range(1, int(max(1, args.combo_size_max)) + 1):
        for combo in itertools.combinations(groups, size):
            static_candidates.append(_build_static_combo(group_monthly, group_meta, combo))

    dynamic_candidates: list[dict[str, Any]] = []
    lookbacks = [int(x.strip()) for x in str(args.dynamic_lookbacks).split(",") if str(x).strip()]
    topks = [int(x.strip()) for x in str(args.dynamic_topk).split(",") if str(x).strip()]
    for lookback, top_k in itertools.product(lookbacks, topks):
        dynamic_candidates.append(
            _build_dynamic_momentum(group_monthly, group_meta, lookback_months=int(lookback), top_k=int(top_k))
        )

    net_profiles = load_net_assumption_profiles(net_cfg_path)
    candidate_rows = []
    selected_group_rows: list[pd.DataFrame] = []
    candidate_monthly_rows: list[pd.DataFrame] = []
    for candidate in [*static_candidates, *dynamic_candidates]:
        row = _evaluate_candidate(candidate=candidate, benchmark_monthly=benchmark_monthly, net_profiles=net_profiles)
        if not row:
            continue
        gross_monthly = pd.to_numeric(candidate["gross_monthly"], errors="coerce").dropna().astype(float)
        turnover_monthly = pd.to_numeric(candidate["turnover_monthly"], errors="coerce").reindex(gross_monthly.index).fillna(0.0).astype(float)
        foreign_share_monthly = pd.to_numeric(candidate["foreign_share_monthly"], errors="coerce").reindex(gross_monthly.index).fillna(0.0).astype(float)
        candidate_monthly_rows.append(
            pd.DataFrame(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "ym": gross_monthly.index.astype(str),
                    "gross_ret": gross_monthly.to_numpy(dtype=float),
                    "turnover": turnover_monthly.to_numpy(dtype=float),
                    "foreign_share": foreign_share_monthly.to_numpy(dtype=float),
                }
            )
        )
        if "selected_groups_csv" in candidate:
            target = outdir / f"{candidate['candidate_id']}_selected_groups.csv"
            candidate["selected_groups_csv"].to_csv(target, index=False)
            row["selected_groups_path"] = str(target)
            selected_group_rows.append(candidate["selected_groups_csv"].assign(candidate_id=str(candidate["candidate_id"])))
        candidate_rows.append(row)

    results_df = pd.DataFrame(candidate_rows).sort_values(
        ["net_blended_ann_return", "gross_ann_return", "net_blended_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if results_df.empty:
        raise SystemExit("no candidate rows produced")

    group_viability = results_df[results_df["kind"] == "group_static_combo"].copy()
    group_viability = group_viability[group_viability["n_groups"] == 1].reset_index(drop=True)

    asset_viability.to_csv(outdir / "asset_viability.csv", index=False)
    group_asset_pool.to_csv(outdir / "group_asset_pool.csv", index=False)
    group_meta.to_csv(outdir / "group_viability_structure.csv", index=False)
    group_monthly.reset_index(names="ym").to_csv(outdir / "group_monthly_returns.csv", index=False)
    benchmark_monthly_df.reset_index(names="ym").to_csv(outdir / "benchmark_monthly_returns.csv", index=False)
    results_df.to_csv(outdir / "candidate_compare.csv", index=False)
    group_viability.to_csv(outdir / "group_viability.csv", index=False)
    if candidate_monthly_rows:
        pd.concat(candidate_monthly_rows, ignore_index=True).to_csv(outdir / "candidate_monthly_returns.csv", index=False)
    if selected_group_rows:
        pd.concat(selected_group_rows, ignore_index=True).to_csv(outdir / "dynamic_selected_groups.csv", index=False)

    top_net = results_df.iloc[0].to_dict()
    top_gross = results_df.sort_values(["gross_ann_return", "gross_sharpe"], ascending=[False, False]).iloc[0].to_dict()
    kept = results_df[results_df["status"] == "keep"].copy()
    top_counter = Counter()
    for text in results_df.head(20)["groups"].astype(str):
        for group in [g.strip() for g in text.split(",") if g.strip()]:
            top_counter[group] += 1
    registry_seed = pd.concat(
        [
            results_df.head(40),
            results_df[results_df["kind"] == "group_dynamic_momentum"].head(10),
            results_df[results_df["n_groups"] == 1].head(10),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["candidate_id"], keep="first")
    research_rows = _build_research_rows(registry_seed, outdir=outdir, summary_path=outdir / "summary.json")
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "inputs": {
            "prices_dir": str(prices_dir),
            "asset_groups_csv": str(asset_groups_csv),
            "asset_metadata_csv": str(asset_metadata_csv),
            "net_assumptions_config": str(net_cfg_path),
            "combo_size_max": int(args.combo_size_max),
            "top_assets_per_group": int(args.top_assets_per_group),
            "min_group_assets": int(args.min_group_assets),
            "min_assets_present": int(args.min_assets_present),
            "dynamic_lookbacks": lookbacks,
            "dynamic_topk": topks,
        },
        "benchmarks": {
            "spy": summarize_return_series(benchmark_monthly["SPY"], periods_per_year=12),
            "sixty_forty": summarize_return_series(benchmark_monthly["sixty_forty"], periods_per_year=12),
            "group_eqw": summarize_return_series(benchmark_monthly["group_eqw"], periods_per_year=12),
        },
        "universe": {
            "groups_total": int(group_monthly.shape[1]),
            "assets_loaded": int(asset_viability.shape[0]),
            "foreign_asset_share": float((asset_viability["jurisdiction"].astype(str) == "foreign").mean()) if not asset_viability.empty else float("nan"),
        },
        "top_candidates": {
            "best_net_blended": top_net,
            "best_gross": top_gross,
        },
        "research_counts": {
            "candidates_total": int(results_df.shape[0]),
            "kept": int((results_df["status"] == "keep").sum()),
            "watch": int((results_df["status"] == "watch").sum()),
            "killed": int((results_df["status"] == "kill").sum()),
            "beat_spy_net": int((results_df["edge_vs_spy_net_total_return"] > 0.0).sum()),
            "beat_spy_gross": int((results_df["edge_vs_spy_gross_total_return"] > 0.0).sum()),
        },
        "insights": [
            f"Melhor candidato liquido: {top_net['candidate_id']} com {float(top_net['net_blended_ann_return']):.4f} a.a. e edge liquido vs SPY de {float(top_net['edge_vs_spy_net_total_return']):.4f}.",
            f"Melhor candidato bruto: {top_gross['candidate_id']} com {float(top_gross['gross_ann_return']):.4f} a.a.",
            f"Top grupos mais frequentes entre os 20 melhores: {dict(top_counter.most_common(6))}.",
            "Comparacao de shadow principal/challenger foi omitida nesta rodada porque a cadeia de retornos acabou de ser corrigida de log para simples e esses baselines precisam ser rerodados.",
        ],
        "official_sources": net_profiles["official_sources"],
        "artifacts": {
            "asset_viability_csv": str(outdir / "asset_viability.csv"),
            "group_asset_pool_csv": str(outdir / "group_asset_pool.csv"),
            "group_viability_csv": str(outdir / "group_viability.csv"),
            "group_monthly_returns_csv": str(outdir / "group_monthly_returns.csv"),
            "benchmark_monthly_returns_csv": str(outdir / "benchmark_monthly_returns.csv"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "candidate_monthly_returns_csv": str(outdir / "candidate_monthly_returns.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir=outdir,
        script="scripts/bench/validation/run_profit_group_methodology_suite.py",
        params={
            "prices_dir": str(prices_dir),
            "asset_groups": str(asset_groups_csv),
            "asset_metadata": str(asset_metadata_csv),
            "net_assumptions_config": str(net_cfg_path),
            "combo_size_max": int(args.combo_size_max),
            "top_assets_per_group": int(args.top_assets_per_group),
            "min_group_assets": int(args.min_group_assets),
            "min_assets_present": int(args.min_assets_present),
            "dynamic_lookbacks": lookbacks,
            "dynamic_topk": topks,
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        gates={"summary_created": True, "candidate_compare_created": True},
    )

    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(outdir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
