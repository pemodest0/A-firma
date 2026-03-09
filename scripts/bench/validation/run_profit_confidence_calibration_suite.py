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
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import (  # noqa: E402
    _blend_allocations,
    _build_confidence_score,
    _human_label,
    _percent_change,
    _result_row,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import _build_breadth_signal  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _rank(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        last = arr[-1]
        return float(np.mean(arr <= last))

    return series.rolling(int(window), min_periods=max(20, int(window) // 4)).apply(_rank, raw=True)


def _logistic_transform(series: pd.Series, center: float, slope: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.5).astype(float)
    return 1.0 / (1.0 + np.exp(-float(slope) * (x - float(center))))


def _continuous_weight(score: pd.Series, *, low: float, high: float, floor: float, gamma: float) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    scaled = ((x - float(floor)) / max(1e-9, 1.0 - float(floor))).clip(lower=0.0, upper=1.0)
    curved = scaled.pow(float(gamma))
    return (float(low) + (float(high) - float(low)) * curved).clip(0.0, 1.0)


def _smooth_weight(
    weight: pd.Series,
    *,
    alpha: float | None = None,
    max_step: float | None = None,
) -> pd.Series:
    out = pd.to_numeric(weight, errors="coerce").ffill().bfill().fillna(0.0).astype(float)
    if alpha is not None:
        out = out.ewm(alpha=float(alpha), adjust=False).mean()
    if max_step is not None:
        capped = out.copy()
        for i in range(1, len(capped)):
            prev = float(capped.iloc[i - 1])
            current = float(capped.iloc[i])
            delta = current - prev
            limit = float(max_step)
            if delta > limit:
                current = prev + limit
            elif delta < -limit:
                current = prev - limit
            capped.iloc[i] = current
        out = capped
    return out.clip(0.0, 1.0)


def _best_by_total_return(candidates: dict[str, StrategyResult]) -> StrategyResult:
    return max(candidates.values(), key=lambda result: (_safe_float(result.net_total_return), _safe_float(result.net_sharpe)))


def _make_bundle_result(
    *,
    candidate_id: str,
    family: str,
    notes: str,
    attack_alloc,
    protect_alloc,
    attack_weight: pd.Series,
):
    bundle = _blend_allocations(
        candidate_id=str(candidate_id),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attack_weight,
    )
    return StrategyResult(
        suite="confidence_calibration",
        candidate_id=str(candidate_id),
        family=str(family),
        benchmark_ticker=bundle.result.benchmark_ticker,
        gross_ret=bundle.result.gross_ret,
        turnover=bundle.result.turnover,
        net_ret=bundle.result.net_ret,
        benchmark_net_ret=bundle.result.benchmark_net_ret,
        net_ann_return=bundle.result.net_ann_return,
        net_total_return=bundle.result.net_total_return,
        net_sharpe=bundle.result.net_sharpe,
        net_max_drawdown=bundle.result.net_max_drawdown,
        edge_vs_benchmark=bundle.result.edge_vs_benchmark,
        avg_turnover_daily=bundle.result.avg_turnover_daily,
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=str(notes),
    )


def _family_piecewise(raw_score: pd.Series, attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    candidates: dict[str, StrategyResult] = {}
    for high in [1.0, 0.95]:
        for medium in [0.65, 0.70, 0.75, 0.80]:
            for low in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for hi_th in [0.60, 0.62, 0.65, 0.68]:
                    for med_th in [0.45, 0.48, 0.50, 0.52, 0.55]:
                        if med_th >= hi_th or low > medium or medium > high:
                            continue
                        weight = pd.Series(low, index=raw_score.index, dtype=float)
                        weight.loc[raw_score >= med_th] = medium
                        weight.loc[raw_score >= hi_th] = high
                        cid = f"piecewise__h{int(high*100)}_m{int(medium*100)}_l{int(low*100)}_t{int(hi_th*100)}_{int(med_th*100)}"
                        candidates[cid] = _make_bundle_result(
                            candidate_id=cid,
                            family="piecewise",
                            notes=f"high={high:.2f};medium={medium:.2f};low={low:.2f};hi_th={hi_th:.2f};med_th={med_th:.2f}",
                            attack_alloc=attack_alloc,
                            protect_alloc=protect_alloc,
                            attack_weight=weight,
                        )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_continuous(raw_score: pd.Series, attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    candidates: dict[str, StrategyResult] = {}
    for high in [1.0, 0.95]:
        for low in [0.10, 0.15, 0.20, 0.25]:
            for floor in [0.20, 0.25, 0.30, 0.35]:
                for gamma in [0.80, 1.00, 1.20, 1.50]:
                    cid = f"continuous__h{int(high*100)}_l{int(low*100)}_f{int(floor*100)}_g{int(gamma*100)}"
                    weight = _continuous_weight(raw_score, low=low, high=high, floor=floor, gamma=gamma)
                    candidates[cid] = _make_bundle_result(
                        candidate_id=cid,
                        family="continuous",
                        notes=f"high={high:.2f};low={low:.2f};floor={floor:.2f};gamma={gamma:.2f}",
                        attack_alloc=attack_alloc,
                        protect_alloc=protect_alloc,
                        attack_weight=weight,
                    )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_transformed(raw_score: pd.Series, attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    candidates: dict[str, StrategyResult] = {}
    transforms: list[tuple[str, pd.Series]] = [
        ("raw", raw_score.clip(0.0, 1.0)),
        ("pct63", _rolling_percentile(raw_score, 63).fillna(raw_score)),
        ("pct126", _rolling_percentile(raw_score, 126).fillna(raw_score)),
        ("logit_c55_s8", _logistic_transform(raw_score, center=0.55, slope=8.0)),
        ("logit_c60_s10", _logistic_transform(raw_score, center=0.60, slope=10.0)),
    ]
    for name, score in transforms:
        for high, medium, low, hi_th, med_th in [
            (1.0, 0.70, 0.20, 0.65, 0.50),
            (1.0, 0.75, 0.15, 0.63, 0.48),
            (0.95, 0.70, 0.20, 0.62, 0.48),
        ]:
            weight = pd.Series(low, index=score.index, dtype=float)
            weight.loc[score >= med_th] = medium
            weight.loc[score >= hi_th] = high
            cid = f"transformed__{name}__h{int(high*100)}_m{int(medium*100)}_l{int(low*100)}"
            candidates[cid] = _make_bundle_result(
                candidate_id=cid,
                family="transformed",
                notes=f"transform={name};high={high:.2f};medium={medium:.2f};low={low:.2f};hi_th={hi_th:.2f};med_th={med_th:.2f}",
                attack_alloc=attack_alloc,
                protect_alloc=protect_alloc,
                attack_weight=weight,
            )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_smoothed(raw_score: pd.Series, attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    candidates: dict[str, StrategyResult] = {}
    base = pd.Series(0.20, index=raw_score.index, dtype=float)
    base.loc[raw_score >= 0.50] = 0.70
    base.loc[raw_score >= 0.65] = 1.0
    for alpha in [0.25, 0.35, 0.50]:
        for max_step in [0.10, 0.15, 0.20, None]:
            weight = _smooth_weight(base, alpha=alpha, max_step=max_step)
            tag = f"a{int(alpha*100)}_d{int((max_step or 99)*100)}"
            cid = f"smoothed__{tag}"
            candidates[cid] = _make_bundle_result(
                candidate_id=cid,
                family="smoothed",
                notes=f"base=100/70/20;alpha={alpha:.2f};max_step={max_step}",
                attack_alloc=attack_alloc,
                protect_alloc=protect_alloc,
                attack_weight=weight,
            )
    return _best_by_total_return(candidates), list(candidates.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra a confiança em volta do campeão de sizing por confiança.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_confidence_calibration_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    context = dict(built["context"])
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]
    baseline_attack = attack_alloc.bundle.result

    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    raw_score = _build_confidence_score(context, breadth_signal, built["sleeve_returns"]["attack"]).clip(0.0, 1.0)

    families: dict[str, tuple[StrategyResult, list[StrategyResult]]] = {
        "piecewise": _family_piecewise(raw_score, attack_alloc, protect_alloc),
        "continuous": _family_continuous(raw_score, attack_alloc, protect_alloc),
        "transformed": _family_transformed(raw_score, attack_alloc, protect_alloc),
        "smoothed": _family_smoothed(raw_score, attack_alloc, protect_alloc),
    }

    winner_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    best_result: StrategyResult | None = None
    for family, (winner, variants) in families.items():
        row = _result_row(winner, baseline=baseline_attack)
        row["family"] = family
        winner_rows.append(row)
        if best_result is None or (_safe_float(winner.net_total_return), _safe_float(winner.net_sharpe)) > (
            _safe_float(best_result.net_total_return),
            _safe_float(best_result.net_sharpe),
        ):
            best_result = winner
        for result in variants:
            row = _result_row(result, baseline=baseline_attack)
            row["family"] = family
            all_rows.append(row)

    winners_df = pd.DataFrame(winner_rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    winners_df.to_csv(outdir / "family_winners.csv", index=False)
    variants_df = pd.DataFrame(all_rows).sort_values(["family", "net_total_return", "net_sharpe"], ascending=[True, False, False]).reset_index(drop=True)
    variants_df.to_csv(outdir / "all_variants.csv", index=False)

    assert best_result is not None
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_attack": _result_row(baseline_attack, baseline=baseline_attack),
        "best_overall": _result_row(best_result, baseline=baseline_attack),
        "family_winners": winner_rows,
        "insights": [
            "A busca foi local, em volta do campeao atual de sizing por confianca.",
            "Foram testadas quatro familias: ajuste discreto dos pesos, curva continua, transformacoes da confianca e suavizacao da troca.",
            "Worth_keeping_alpha marca so as variantes que melhoraram o lucro final contra o ataque atual.",
        ],
        "artifacts": {
            "family_winners_csv": str(outdir / "family_winners.csv"),
            "all_variants_csv": str(outdir / "all_variants.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_confidence_calibration_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "family_winners_csv": str(outdir / "family_winners.csv"),
            "all_variants_csv": str(outdir / "all_variants.csv"),
        },
        extra={
            "suite": "profit_confidence_calibration_suite",
            "best_candidate_id": str(best_result.candidate_id),
            "best_candidate_label": _human_label(str(best_result.candidate_id)),
            "best_total_return": _safe_float(best_result.net_total_return),
            "best_ann_return": _safe_float(best_result.net_ann_return),
            "best_sharpe": _safe_float(best_result.net_sharpe),
            "best_max_drawdown": _safe_float(best_result.net_max_drawdown),
            "baseline_total_return": _safe_float(baseline_attack.net_total_return),
        },
        repo_root=ROOT,
    )
    print(outdir)


if __name__ == "__main__":
    main()
