#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio.exogenous_features import (  # noqa: E402
    adjust_confidence_with_feature,
    build_exogenous_feature_panel,
    feature_spectral_extremes,
    load_market_series,
)
from engine.structural.covariance_estimators import estimate_corr  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _build_confidence_score, _blend_allocations, _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import (  # noqa: E402
    _build_attack_allocation,
    _make_crypto_bundle,
    _result_row,
    _weight_from_current_champion,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import _build_breadth_signal  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_base_fast_entry(*, context: dict[str, Any], protect_alloc):
    major8 = context["crypto_tiers"]["crypto_major8"]
    base_crypto_bundle = _make_crypto_bundle(
        candidate_id="major8_mom_total_lb21_rb07_k1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
    )
    attack_alloc = _build_attack_allocation(
        candidate_id="entry_fast14_exit63_m2_h0__attack",
        context=context,
        crypto_bundle=base_crypto_bundle,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    attack_returns = pd.concat(
        {
            "crypto": pd.to_numeric(base_crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(context["equity_attack"].result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    raw_score = _build_confidence_score(context, breadth_signal, attack_returns).clip(0.0, 1.0)
    base_weight = _weight_from_current_champion(raw_score)
    bundle = _blend_allocations(
        candidate_id="baseline_fast_entry",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=base_weight,
    )
    return {
        "crypto_bundle": base_crypto_bundle,
        "attack_alloc": attack_alloc,
        "attack_returns": attack_returns,
        "breadth_signal": breadth_signal,
        "raw_score": raw_score,
        "weight": base_weight,
        "result": bundle.result,
    }


def _monthly_spectral_panel(*, context: dict[str, Any], prices_dir: Path) -> pd.DataFrame:
    crypto_cols = [c for c in ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "LTC-USD", "TRX-USD"] if c in context["crypto_returns"].columns]
    macro_tickers = ["SPY", "HYG", "LQD", "TLT", "SHY", "UUP", "^VIX"]
    macro_returns: dict[str, pd.Series] = {}
    base_idx = context["crypto_returns"].index
    for ticker in macro_tickers:
        _, ret = load_market_series(prices_dir, ticker, base_idx)
        if ret.notna().sum() > 200:
            macro_returns[ticker] = ret
    mat = pd.concat(
        [context["crypto_returns"][crypto_cols]] + [pd.DataFrame(macro_returns, index=base_idx)],
        axis=1,
        sort=False,
    ).dropna(axis=1, how="all")
    idx = mat.index
    rows: list[dict[str, Any]] = []
    step = 21
    window = 120
    for end in range(window, len(idx), step):
        sub = mat.iloc[end - window : end].dropna(axis=1, thresh=max(80, window // 2))
        sub = sub.fillna(0.0)
        if sub.shape[1] < 4:
            continue
        corr = estimate_corr(sub.to_numpy(dtype=float), method="ledoit_wolf")
        eig = np.linalg.eigvalsh(corr)
        eig = np.sort(np.real(eig))[::-1]
        total = float(np.sum(eig))
        if total <= 0:
            continue
        p = eig / total
        upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
        rows.append(
            {
                "date": idx[end - 1],
                "lambda1": float(eig[0]),
                "p1": float(p[0]),
                "deff": float(1.0 / np.sum(np.square(p))),
                "avg_abs_corr": float(np.abs(corr[upper]).mean()) if int(upper.sum()) > 0 else float("nan"),
                "n_assets": int(sub.shape[1]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date").sort_index()
    return out


def _ecosystem_compare(*, spectral_panel: pd.DataFrame, baseline_weight: pd.Series, variant_weight: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if spectral_panel.empty:
        return pd.DataFrame(rows)
    for label, weight in [("baseline", baseline_weight), ("variant", variant_weight)]:
        idx = spectral_panel.index.intersection(weight.index)
        w = pd.to_numeric(weight.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        s = spectral_panel.reindex(idx)
        full = s[w >= 0.95]
        mid = s[(w >= 0.50) & (w < 0.95)]
        low = s[w < 0.50]
        for state, frame in [("full_attack", full), ("mid_attack", mid), ("protected", low)]:
            if frame.empty:
                continue
            rows.append(
                {
                    "candidate": label,
                    "state": state,
                    "p1": float(pd.to_numeric(frame["p1"], errors="coerce").mean()),
                    "deff": float(pd.to_numeric(frame["deff"], errors="coerce").mean()),
                    "avg_abs_corr": float(pd.to_numeric(frame["avg_abs_corr"], errors="coerce").mean()),
                    "lambda1": float(pd.to_numeric(frame["lambda1"], errors="coerce").mean()),
                    "n_points": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def _variant_bundle(
    *,
    candidate_id: str,
    feature_panel: pd.DataFrame,
    feature_col: str,
    feature_mode: str,
    feature_weight: float,
    base_score: pd.Series,
    attack_alloc,
    protect_alloc,
) -> tuple[StrategyResult, pd.Series]:
    adj_score = adjust_confidence_with_feature(
        base_score=base_score,
        feature=feature_panel[feature_col],
        mode=feature_mode,
        weight=feature_weight,
    )
    attack_weight = _weight_from_current_champion(adj_score)
    bundle = _blend_allocations(
        candidate_id=str(candidate_id),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attack_weight,
    )
    result = replace(bundle.result, notes=f"{feature_col}:{feature_mode}:{feature_weight:.2f}")
    return result, attack_weight


def _combo_bundle(
    *,
    candidate_id: str,
    panel: pd.DataFrame,
    base_score: pd.Series,
    attack_alloc,
    protect_alloc,
    adjustments: list[tuple[str, str, float]],
) -> tuple[StrategyResult, pd.Series]:
    score = base_score.copy()
    for col, mode, weight in adjustments:
        score = adjust_confidence_with_feature(
            base_score=score,
            feature=panel[col],
            mode=mode,
            weight=weight,
        )
    attack_weight = _weight_from_current_champion(score)
    bundle = _blend_allocations(
        candidate_id=str(candidate_id),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attack_weight,
    )
    result = replace(bundle.result, notes=";".join(f"{c}:{m}:{w:.2f}" for c, m, w in adjustments))
    return result, attack_weight


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara sinais exogenos cripto+macro sobre o campeao atual do ataque.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=1000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_exogenous_feature_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    context = dict(built["context"])
    protect_alloc = built["allocations"]["baseline_guard"]
    base = _build_base_fast_entry(context=context, protect_alloc=protect_alloc)

    exo = build_exogenous_feature_panel(
        prices_dir=prices_dir,
        crypto_returns=context["crypto_returns"],
        crypto_prices=context["crypto_prices"],
        benchmark_crypto=str(args.benchmark_crypto),
        macro_index=base["raw_score"].index,
    )

    specs = [
        ("exo_vix", "VIX", "penalty", 0.16),
        ("exo_credit_spreads", "credit_spreads", "penalty", 0.14),
        ("exo_rates", "rates", "penalty", 0.10),
        ("exo_dollar", "dollar", "penalty", 0.10),
        ("exo_liquidity", "liquidity", "boost", 0.12),
        ("exo_funding", "funding", "penalty", 0.12),
        ("exo_open_interest", "open_interest", "penalty", 0.10),
        ("exo_liquidation", "liquidation", "penalty", 0.14),
        ("exo_btc_dominance", "btc_dominance", "penalty", 0.10),
        ("exo_breadth", "breadth", "breadth_boost", 0.16),
    ]
    results: dict[str, StrategyResult] = {"baseline_fast_entry": base["result"]}
    weights: dict[str, pd.Series] = {"baseline_fast_entry": base["weight"]}
    rows = [
        _result_row(
            base["result"],
            baseline=base["result"],
            family="baseline",
            label="Campeão atual com entrada rápida",
        )
    ]

    for cid, col, mode, weight in specs:
        result, attack_weight = _variant_bundle(
            candidate_id=cid,
            feature_panel=exo.panel,
            feature_col=col,
            feature_mode=mode,
            feature_weight=weight,
            base_score=base["raw_score"],
            attack_alloc=base["attack_alloc"],
            protect_alloc=protect_alloc,
        )
        results[cid] = result
        weights[cid] = attack_weight
        rows.append(_result_row(result, baseline=base["result"], family="single_exogenous", label=cid))

    combo_configs = {
        "exo_macro_combo": [
            ("VIX", "penalty", 0.10),
            ("credit_spreads", "penalty", 0.08),
            ("rates", "penalty", 0.06),
            ("dollar", "penalty", 0.06),
            ("liquidity", "boost", 0.08),
        ],
        "exo_crypto_combo": [
            ("funding", "penalty", 0.08),
            ("open_interest", "penalty", 0.06),
            ("liquidation", "penalty", 0.10),
            ("btc_dominance", "penalty", 0.06),
            ("breadth", "breadth_boost", 0.10),
        ],
        "exo_all_combo": [
            ("VIX", "penalty", 0.08),
            ("credit_spreads", "penalty", 0.06),
            ("dollar", "penalty", 0.04),
            ("funding", "penalty", 0.06),
            ("liquidation", "penalty", 0.08),
            ("breadth", "breadth_boost", 0.10),
            ("liquidity", "boost", 0.06),
        ],
    }
    for cid, adjustments in combo_configs.items():
        result, attack_weight = _combo_bundle(
            candidate_id=cid,
            panel=exo.panel,
            base_score=base["raw_score"],
            attack_alloc=base["attack_alloc"],
            protect_alloc=protect_alloc,
            adjustments=adjustments,
        )
        results[cid] = result
        weights[cid] = attack_weight
        rows.append(_result_row(result, baseline=base["result"], family="combo_exogenous", label=cid))

    compare_df = pd.DataFrame(rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    best_row = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    best_id = str(best_row.get("candidate_id", "baseline_fast_entry"))
    best_result = results[best_id]
    best_weight = weights[best_id]

    yearbook_rows: list[dict[str, Any]] = []
    for cid in ["baseline_fast_entry", best_id]:
        yearbook_rows.extend(_calendar_rows(result=results[cid], capital_brl=float(args.capital_brl)))
    yearbook_df = pd.DataFrame(yearbook_rows)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    spectral_panel = _monthly_spectral_panel(context=context, prices_dir=prices_dir)
    spectral_panel.to_csv(outdir / "spectral_panel.csv", index=True)
    spectral_effects = feature_spectral_extremes(
        feature_panel=exo.panel,
        spectral_panel=spectral_panel,
        feature_cols=exo.crypto_columns + exo.macro_columns + ["exogenous_risk"],
    )
    spectral_effects.to_csv(outdir / "feature_spectral_effects.csv", index=False)
    ecosystem_df = _ecosystem_compare(
        spectral_panel=spectral_panel,
        baseline_weight=weights["baseline_fast_entry"],
        variant_weight=best_weight,
    )
    ecosystem_df.to_csv(outdir / "ecosystem_compare.csv", index=False)
    exo.panel.to_csv(outdir / "exogenous_panel.csv", index=True)

    research_rows = [
        _research_row(base["result"], outdir=ROOT / "results" / "validation", status="keep", methodology="attack_fast_entry_baseline", label="Ataque rápido atual"),
    ]
    if best_id != "baseline_fast_entry":
        research_rows.append(
            _research_row(best_result, outdir=ROOT / "results" / "validation", status="watch", methodology="exogenous_feature_overlay", label=f"Overlay exógeno: {best_id}")
        )
    pd.DataFrame(research_rows).to_csv(outdir / "research_rows.csv", index=False)

    summary = {
        "suite": "profit_exogenous_feature_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": rows[0],
        "best_overall": best_row,
        "best_candidate_id": best_id,
        "worth_keeping": bool(best_id != "baseline_fast_entry" and _safe_float(best_result.net_total_return) > _safe_float(base["result"].net_total_return)),
        "feature_columns": list(exo.panel.columns),
        "spectral_panel_rows": int(len(spectral_panel)),
        "files": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "feature_spectral_effects_csv": str(outdir / "feature_spectral_effects.csv"),
            "ecosystem_compare_csv": str(outdir / "ecosystem_compare.csv"),
            "exogenous_panel_csv": str(outdir / "exogenous_panel.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_exogenous_feature_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "capital_brl": float(args.capital_brl),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "spectral_panel_csv": str(outdir / "spectral_panel.csv"),
            "feature_spectral_effects_csv": str(outdir / "feature_spectral_effects.csv"),
            "ecosystem_compare_csv": str(outdir / "ecosystem_compare.csv"),
            "exogenous_panel_csv": str(outdir / "exogenous_panel.csv"),
        },
        extra={
            "suite": "profit_exogenous_feature_suite",
            "best_candidate_id": str(best_result.candidate_id),
            "best_total_return": _safe_float(best_result.net_total_return),
            "best_ann_return": _safe_float(best_result.net_ann_return),
            "best_sharpe": _safe_float(best_result.net_sharpe),
            "best_max_drawdown": _safe_float(best_result.net_max_drawdown),
            "baseline_candidate_id": str(base["result"].candidate_id),
            "baseline_total_return": _safe_float(base["result"].net_total_return),
        },
        repo_root=ROOT,
    )


if __name__ == "__main__":
    main()
