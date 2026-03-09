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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_attack_validation_suite import _bootstrap_returns  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    StrategyBundle,
    _evaluate_allocation_candidate,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.ops.run_profit_shadow_suite import _load_price_returns  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _human_label(candidate_id: str) -> str:
    mapping = {
        "meta_major8_eq_a2r1": "Modo principal",
        "alpha_attack_major8_equity25": "Modo ataque",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda",
        "pure_crypto_attack": "Sem ações",
        "pure_equity_attack": "Sem cripto",
        "blend_half_attack": "Sem meta-switch",
        "attack_no_matrix": "Sem matriz",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _weights_frame(index: pd.Index, *, crypto: float, equity: float, cash: float) -> pd.DataFrame:
    return pd.DataFrame({"crypto": float(crypto), "equity": float(equity), "cash": float(cash)}, index=index, dtype=float)


def _bundle_from_sleeves(
    *,
    candidate_id: str,
    family: str,
    weights: pd.DataFrame,
    returns_frame: pd.DataFrame,
    benchmark_ret: pd.Series,
    profile,
) -> StrategyBundle:
    idx = returns_frame.index.intersection(benchmark_ret.index).sort_values()
    w = weights.reindex(idx).ffill().fillna(0.0).astype(float)
    returns = returns_frame.reindex(idx).fillna(0.0).astype(float)
    return _evaluate_allocation_candidate(
        candidate_id=str(candidate_id),
        family=str(family),
        weights=w,
        crypto_ret=returns["crypto"],
        equity_ret=returns["equity"],
        benchmark_ret=benchmark_ret.reindex(idx).fillna(0.0).astype(float),
        profile=profile,
        benchmark_profile=profile,
        notes="historical closure bundle",
    )


def _compound_periods(result: StrategyBundle, *, freq: str) -> pd.DataFrame:
    ret = pd.to_numeric(result.result.net_ret, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(result.result.benchmark_net_ret, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    if ret.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"ret": ret, "bench": bench}, index=pd.to_datetime(ret.index))
    rows: list[dict[str, Any]] = []
    for period, sub in df.groupby(df.index.to_period(freq)):
        strat_total = float(np.prod(1.0 + sub["ret"].to_numpy(dtype=float)) - 1.0)
        bench_total = float(np.prod(1.0 + sub["bench"].to_numpy(dtype=float)) - 1.0)
        rows.append(
            {
                "candidate_id": str(result.result.candidate_id),
                "candidate_label": _human_label(result.result.candidate_id),
                "period": str(period),
                "strategy_total_return": strat_total,
                "benchmark_total_return": bench_total,
                "alpha_total_return": strat_total - bench_total,
                "won_vs_benchmark": int(strat_total > bench_total),
                "lost_vs_benchmark": int(strat_total < bench_total),
            }
        )
    return pd.DataFrame(rows)


def _longest_negative_streak(alpha: pd.Series) -> int:
    longest = 0
    current = 0
    for value in pd.to_numeric(alpha, errors="coerce").fillna(0.0).astype(float):
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _frozen_row(bundle: StrategyBundle, *, capital_brl: float) -> dict[str, Any]:
    yearbook = pd.DataFrame(_calendar_rows(result=bundle.result, capital_brl=float(capital_brl)))
    monthly = _compound_periods(bundle, freq="M")
    quarterly = _compound_periods(bundle, freq="Q")
    ret = pd.to_numeric(bundle.result.net_ret, errors="coerce").dropna().astype(float)
    operations = int((pd.to_numeric(bundle.result.turnover, errors="coerce").reindex(ret.index).fillna(0.0).abs() > 1e-8).sum())
    best_year = yearbook.sort_values("profit_brl", ascending=False).head(1)
    worst_year = yearbook.sort_values("profit_brl", ascending=True).head(1)
    return {
        "candidate_id": str(bundle.result.candidate_id),
        "candidate_label": _human_label(bundle.result.candidate_id),
        "capital_final_brl": float(capital_brl * (1.0 + float(bundle.result.net_total_return))),
        "profit_total_brl": float(capital_brl * float(bundle.result.net_total_return)),
        "net_ann_return": float(bundle.result.net_ann_return),
        "net_total_return": float(bundle.result.net_total_return),
        "net_sharpe": float(bundle.result.net_sharpe),
        "net_max_drawdown": float(bundle.result.net_max_drawdown),
        "operation_days_total": operations,
        "years_positive": int((yearbook["profit_brl"] > 0.0).sum()) if not yearbook.empty else 0,
        "years_negative": int((yearbook["profit_brl"] < 0.0).sum()) if not yearbook.empty else 0,
        "months_won": int(monthly["won_vs_benchmark"].sum()) if not monthly.empty else 0,
        "months_lost": int(monthly["lost_vs_benchmark"].sum()) if not monthly.empty else 0,
        "quarters_won": int(quarterly["won_vs_benchmark"].sum()) if not quarterly.empty else 0,
        "quarters_lost": int(quarterly["lost_vs_benchmark"].sum()) if not quarterly.empty else 0,
        "longest_monthly_underperf_streak": _longest_negative_streak(monthly["alpha_total_return"]) if not monthly.empty else 0,
        "longest_quarterly_underperf_streak": _longest_negative_streak(quarterly["alpha_total_return"]) if not quarterly.empty else 0,
        "best_year": int(best_year.iloc[0]["year"]) if not best_year.empty else None,
        "best_year_profit_brl": float(best_year.iloc[0]["profit_brl"]) if not best_year.empty else None,
        "worst_year": int(worst_year.iloc[0]["year"]) if not worst_year.empty else None,
        "worst_year_profit_brl": float(worst_year.iloc[0]["profit_brl"]) if not worst_year.empty else None,
    }


def _bootstrap_summary(bundle: StrategyBundle, *, seed: int) -> dict[str, Any]:
    ret = pd.to_numeric(bundle.result.net_ret, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(bundle.result.benchmark_net_ret, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    if ret.empty:
        return {"status": "empty"}
    monthly = pd.DataFrame({"ret": ret, "bench": bench}, index=pd.to_datetime(ret.index)).groupby(pd.to_datetime(ret.index).to_period("M")).agg(
        ret=("ret", lambda x: float(np.prod(1.0 + pd.Series(x).to_numpy(dtype=float)) - 1.0)),
        bench=("bench", lambda x: float(np.prod(1.0 + pd.Series(x).to_numpy(dtype=float)) - 1.0)),
    )
    rng = np.random.default_rng(int(seed))
    daily_boot = _bootstrap_returns(
        returns=ret.to_numpy(dtype=float),
        benchmark=bench.to_numpy(dtype=float),
        periods_per_year=252.0,
        rng=rng,
        n_iter=800,
        sample_len=int(len(ret)),
        block_size=21,
    )
    monthly_boot = _bootstrap_returns(
        returns=pd.to_numeric(monthly["ret"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        benchmark=pd.to_numeric(monthly["bench"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        periods_per_year=12.0,
        rng=np.random.default_rng(int(seed) + 17),
        n_iter=800,
        sample_len=int(len(monthly)),
        block_size=3,
    )
    return {"daily": daily_boot, "monthly": monthly_boot}


def _confidence_rows(candidate_id: str, boot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in ["daily", "monthly"]:
        b = boot.get(horizon) or {}
        if str(b.get("status")) != "ok":
            continue
        rows.extend(
            [
                {
                    "candidate_id": str(candidate_id),
                    "candidate_label": _human_label(candidate_id),
                    "horizon": str(horizon),
                    "scenario": "ruim",
                    "total_return": b.get("strategy_total_return_p05"),
                    "ann_return": b.get("strategy_ann_return_p05"),
                    "drawdown": b.get("strategy_mdd_p05"),
                },
                {
                    "candidate_id": str(candidate_id),
                    "candidate_label": _human_label(candidate_id),
                    "horizon": str(horizon),
                    "scenario": "central",
                    "total_return": b.get("strategy_total_return_p50"),
                    "ann_return": b.get("strategy_ann_return_p50"),
                    "drawdown": b.get("strategy_mdd_p50"),
                },
                {
                    "candidate_id": str(candidate_id),
                    "candidate_label": _human_label(candidate_id),
                    "horizon": str(horizon),
                    "scenario": "bom",
                    "total_return": b.get("strategy_total_return_p95"),
                    "ann_return": b.get("strategy_ann_return_p95"),
                    "drawdown": b.get("strategy_mdd_p95"),
                },
            ]
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fechamento historico: finalistas, month/quarter tournament, bootstrap, ablation, yearbook e confianca.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_historical_closure_suite")
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

    attack_alloc = built["allocations"]["attack"]
    base_alloc = built["allocations"]["baseline"]
    attack_sleeves = built["sleeve_returns"]["attack"]
    benchmark_ret = attack_alloc.bundle.benchmark_gross_ret

    pure_crypto_attack = _bundle_from_sleeves(
        candidate_id="pure_crypto_attack",
        family="ablation",
        weights=_weights_frame(attack_sleeves.index, crypto=1.0, equity=0.0, cash=0.0),
        returns_frame=attack_sleeves,
        benchmark_ret=benchmark_ret,
        profile=attack_alloc.bundle.profile,
    )
    pure_equity_attack = _bundle_from_sleeves(
        candidate_id="pure_equity_attack",
        family="ablation",
        weights=_weights_frame(attack_sleeves.index, crypto=0.0, equity=1.0, cash=0.0),
        returns_frame=attack_sleeves,
        benchmark_ret=benchmark_ret,
        profile=attack_alloc.bundle.profile,
    )
    blend_half_attack = _bundle_from_sleeves(
        candidate_id="blend_half_attack",
        family="ablation",
        weights=_weights_frame(attack_sleeves.index, crypto=0.5, equity=0.5, cash=0.0),
        returns_frame=attack_sleeves,
        benchmark_ret=benchmark_ret,
        profile=attack_alloc.bundle.profile,
    )

    spy_ret = _load_price_returns(prices_dir, str(args.benchmark_equity)).rename("SPY")
    idx = attack_sleeves.index.intersection(spy_ret.index).sort_values()
    spy_returns_frame = pd.DataFrame({"crypto": 0.0, "equity": spy_ret.reindex(idx).fillna(0.0).astype(float)}, index=idx, dtype=float)
    spy_equity_bundle = _bundle_from_sleeves(
        candidate_id="spy_equity_simple",
        family="ablation",
        weights=_weights_frame(idx, crypto=0.0, equity=1.0, cash=0.0),
        returns_frame=spy_returns_frame,
        benchmark_ret=benchmark_ret.reindex(idx).fillna(0.0).astype(float),
        profile=attack_alloc.bundle.profile,
    )
    btc_prices = _load_price_returns(prices_dir, str(args.benchmark_crypto))
    spy_prices = _load_price_returns(prices_dir, str(args.benchmark_equity))
    attack_no_matrix = _build_alpha_meta_allocation_bundle(
        candidate_id="attack_no_matrix",
        crypto_bundle=pure_crypto_attack,
        equity_bundle=spy_equity_bundle,
        btc_prices=btc_prices.add(1.0).cumprod(),
        spy_prices=spy_prices.add(1.0).cumprod(),
        profile=attack_alloc.bundle.profile,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    ).bundle

    finalists: list[StrategyBundle] = [
        built["attack"],
        built["baseline"],
        built["attack_guard"],
        built["baseline_guard"],
    ]
    ablations: list[StrategyBundle] = [
        pure_crypto_attack,
        pure_equity_attack,
        blend_half_attack,
        attack_no_matrix,
    ]

    frozen_rows = [_frozen_row(bundle, capital_brl=float(args.capital_brl)) for bundle in finalists]
    frozen_df = pd.DataFrame(frozen_rows).sort_values(["profit_total_brl", "net_ann_return"], ascending=[False, False]).reset_index(drop=True)
    frozen_df.to_csv(outdir / "frozen_overview.csv", index=False)

    month_rows: list[pd.DataFrame] = []
    quarter_rows: list[pd.DataFrame] = []
    yearbook_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    for pos, bundle in enumerate(finalists, start=1):
        monthly = _compound_periods(bundle, freq="M")
        quarterly = _compound_periods(bundle, freq="Q")
        if not monthly.empty:
            month_rows.append(monthly)
        if not quarterly.empty:
            quarter_rows.append(quarterly)
        yearbook_rows.extend(_calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl)))
        boot = _bootstrap_summary(bundle, seed=23 + pos * 11)
        bootstrap_rows.append(
            {
                "candidate_id": str(bundle.result.candidate_id),
                "candidate_label": _human_label(bundle.result.candidate_id),
                "daily_prob_total_positive": ((boot.get("daily") or {}).get("prob_strategy_total_positive")),
                "daily_prob_beats_benchmark": ((boot.get("daily") or {}).get("prob_strategy_beats_benchmark")),
                "daily_prob_mdd_worse_35": ((boot.get("daily") or {}).get("prob_mdd_worse_than_35pct")),
                "monthly_prob_total_positive": ((boot.get("monthly") or {}).get("prob_strategy_total_positive")),
                "monthly_prob_beats_benchmark": ((boot.get("monthly") or {}).get("prob_strategy_beats_benchmark")),
                "monthly_prob_mdd_worse_35": ((boot.get("monthly") or {}).get("prob_mdd_worse_than_35pct")),
            }
        )
        confidence_rows.extend(_confidence_rows(bundle.result.candidate_id, boot))

    monthly_df = pd.concat(month_rows, ignore_index=True).sort_values(["period", "alpha_total_return"], ascending=[True, False]).reset_index(drop=True)
    quarterly_df = pd.concat(quarter_rows, ignore_index=True).sort_values(["period", "alpha_total_return"], ascending=[True, False]).reset_index(drop=True)
    yearbook_df = pd.DataFrame(yearbook_rows).sort_values(["year", "profit_brl"], ascending=[True, False]).reset_index(drop=True)
    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(["monthly_prob_beats_benchmark", "daily_prob_beats_benchmark"], ascending=[False, False]).reset_index(drop=True)
    confidence_df = pd.DataFrame(confidence_rows)

    monthly_df.to_csv(outdir / "monthly_tournament.csv", index=False)
    quarterly_df.to_csv(outdir / "quarterly_tournament.csv", index=False)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)
    bootstrap_df.to_csv(outdir / "bootstrap_compare.csv", index=False)
    confidence_df.to_csv(outdir / "confidence_bands.csv", index=False)

    ablation_rows: list[dict[str, Any]] = []
    for bundle in [built["attack"], pure_crypto_attack, pure_equity_attack, blend_half_attack, attack_no_matrix, built["attack_guard"]]:
        ablation_rows.append(
            {
                "candidate_id": str(bundle.result.candidate_id),
                "candidate_label": _human_label(bundle.result.candidate_id),
                "net_ann_return": float(bundle.result.net_ann_return),
                "net_total_return": float(bundle.result.net_total_return),
                "net_max_drawdown": float(bundle.result.net_max_drawdown),
                "edge_vs_benchmark": float(bundle.result.edge_vs_benchmark),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values(["net_total_return", "net_ann_return"], ascending=[False, False]).reset_index(drop=True)
    ablation_df.to_csv(outdir / "ablation_compare.csv", index=False)

    research_rows = [
        _research_row(built["attack"].result, outdir=outdir, status="keep", methodology="historical_closure_attack", label="Modo ataque validado no fechamento histórico"),
        _research_row(built["baseline"].result, outdir=outdir, status="keep", methodology="historical_closure_balanced", label="Modo principal validado no fechamento histórico"),
        _research_row(built["attack_guard"].result, outdir=outdir, status="watch", methodology="historical_closure_attack_guard", label="Modo ataque com proteção"),
        _research_row(built["baseline_guard"].result, outdir=outdir, status="watch", methodology="historical_closure_balanced_guard", label="Modo principal com proteção"),
    ]
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_profit_mode": frozen_df.iloc[0].to_dict() if not frozen_df.empty else {},
        "best_bootstrap_mode": bootstrap_df.iloc[0].to_dict() if not bootstrap_df.empty else {},
        "ablation_baseline": ablation_df.to_dict(orient="records"),
        "key_answers": {
            "historical_gain_exists": bool(not frozen_df.empty and float(frozen_df.iloc[0]["profit_total_brl"]) > 0.0),
            "most_profitable_mode": str(frozen_df.iloc[0]["candidate_label"]) if not frozen_df.empty else "",
            "most_consistent_mode_monthly": str(
                monthly_df.groupby("candidate_label")["won_vs_benchmark"].sum().sort_values(ascending=False).index[0]
            )
            if not monthly_df.empty
            else "",
            "mode_with_best_confidence": str(bootstrap_df.iloc[0]["candidate_label"]) if not bootstrap_df.empty else "",
        },
        "insights": [
            "O shadow histórico congelado responde se o ganho realmente existiu sem reotimizar no meio do caminho.",
            "O torneio mensal e trimestral mostra se o lucro é contínuo ou se vem de poucos surtos fortes.",
            "O bootstrap por blocos mede se o ganho parece robusto ou se depende demais de uma ordem específica do histórico.",
            "A ablação mostra o que acontece quando se tira cripto, ações, meta-switch, guarda ou a perna inteligente de ações.",
            "A faixa de confiança resume cenário ruim, central e bom para cada finalista sem prometer futuro.",
        ],
        "artifacts": {
            "frozen_overview_csv": str(outdir / "frozen_overview.csv"),
            "monthly_tournament_csv": str(outdir / "monthly_tournament.csv"),
            "quarterly_tournament_csv": str(outdir / "quarterly_tournament.csv"),
            "bootstrap_compare_csv": str(outdir / "bootstrap_compare.csv"),
            "ablation_compare_csv": str(outdir / "ablation_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "confidence_bands_csv": str(outdir / "confidence_bands.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_historical_closure_suite.py",
        params={"capital_brl": args.capital_brl, "benchmark_crypto": args.benchmark_crypto, "benchmark_equity": args.benchmark_equity},
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
