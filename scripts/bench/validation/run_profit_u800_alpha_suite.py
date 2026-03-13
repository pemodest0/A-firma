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
    _build_promoted_attack_confidence_score,
    _blend_allocation_bundles,
    _confidence_weight_from_score,
)
from scripts.bench.validation.run_profit_equity_improvement_suite import (  # noqa: E402
    _equity_trailing_switch_bundle,
    _load_equity_universe,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    _result_row,
    _safe_float,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
)
from scripts.bench.validation.run_profit_universe_resilience_suite import (  # noqa: E402
    _selection_frequency_for_crypto_rule,
)
from engine.portfolio.exogenous_features import adjust_confidence_with_feature  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _crypto_bundle(
    *,
    candidate_id: str,
    allowed_tickers: list[str],
    context: dict[str, Any],
) -> StrategyBundle:
    result = _simulate_asset_rule(
        candidate_id=candidate_id,
        family="u800_attack_crypto",
        allowed_tickers=list(allowed_tickers),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        asset_table=context["crypto_assets"],
        benchmark_ticker=str(context["benchmark_crypto"]),
        fallback_ticker=str(context["benchmark_crypto"]),
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=False,
        skip_recent_days=0,
        trailing_stop_dd=None,
        hard_stop_loss=None,
        stop_to_cash=True,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    if result is None:
        raise SystemExit(f"falha ao simular sleeve cripto {candidate_id}")
    benchmark = (
        pd.to_numeric(context["crypto_returns"][str(context["benchmark_crypto"])], errors="coerce")
        .reindex(result.gross_ret.index)
        .fillna(0.0)
        .astype(float)
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )


def _attack_from_parts(
    *,
    candidate_id: str,
    label: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    baseline_guard_alloc: Any,
    context: dict[str, Any],
    exogenous_weight: float = 0.14,
) -> StrategyBundle:
    raw_attack = _build_alpha_meta_allocation_bundle(
        candidate_id=f"{candidate_id}__raw",
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_bundle,
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    sleeve_returns = (
        pd.concat(
            {
                "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(equity_bundle.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        )
        .dropna(how="all")
    )
    score = _build_promoted_attack_confidence_score(
        {
            "btc_prices": context["btc_prices"],
            "spy_prices": context["spy_prices"],
            "regime_series": context["regime_series"],
        },
        sleeve_returns,
    )
    score = adjust_confidence_with_feature(
        base_score=score,
        feature=context["exogenous_panel"].get("liquidation"),
        mode="penalty",
        weight=float(exogenous_weight),
    )
    attack = _blend_allocation_bundles(
        candidate_id=candidate_id,
        notes=label,
        attack_alloc=raw_attack,
        protect_alloc=baseline_guard_alloc,
        attack_weight=_confidence_weight_from_score(score),
    )
    return attack.bundle


def _build_u800_equity_candidates(
    *,
    prices_dir: Path,
    asset_groups: Path,
    asset_metadata: Path,
    benchmark_ticker: str,
    profile: Any,
) -> dict[str, StrategyBundle]:
    asset_table, returns, prices, group_map = _load_equity_universe(
        prices_dir=prices_dir,
        asset_groups=asset_groups,
        asset_metadata=asset_metadata,
        benchmark_ticker=benchmark_ticker,
    )
    spy_prices = pd.to_numeric(prices[str(benchmark_ticker)], errors="coerce")
    # Reuse the same structural regime already materialized in the main candidate builder.
    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=ROOT / "data" / "asset_groups_crypto_top_liquid_plus.csv",
        crypto_meta=ROOT / "data" / "asset_metadata_crypto_top_liquid_plus.csv",
        equity_groups=asset_groups,
        equity_meta=asset_metadata,
        benchmark_crypto="BTC-USD",
        benchmark_equity=benchmark_ticker,
    )
    regime_series = built["context"]["regime_series"]

    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="equity_v2__slow189__g4__a1",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=group_map,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=profile,
        benchmark_profile=profile,
    )
    eq_r1 = _simulate_equity_group_sleeve_v3(
        candidate_id="equity_v3__slow189__g3__a2__br35__cap40",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=group_map,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.35,
        max_group_weight=0.40,
        profile=profile,
        benchmark_profile=profile,
    )
    eq_r3 = _simulate_equity_group_sleeve_v3(
        candidate_id="equity_v3__slow252__g3__a2__br30__cap45",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=group_map,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=63,
        group_lookback_slow=252,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.30,
        max_group_weight=0.45,
        profile=profile,
        benchmark_profile=profile,
    )
    if eq_a2 is None or eq_r1 is None or eq_r3 is None:
        raise SystemExit("falha ao reconstruir sleeves do universo 800")

    return {
        "a2r1": _equity_trailing_switch_bundle(
            candidate_id="equity_meta_search__trail_switch__a2__r1",
            aggressive_bundle=eq_a2,
            robust_bundle=eq_r1,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="trail_switch",
        ),
        "a2r3": _equity_trailing_switch_bundle(
            candidate_id="equity_meta_search__trail_switch__a2__r3",
            aggressive_bundle=eq_a2,
            robust_bundle=eq_r3,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="trail_switch",
        ),
        "regime_blend_a2r3": _equity_trailing_switch_bundle(
            candidate_id="equity_meta_search__regime_blend__a2__r3",
            aggressive_bundle=eq_a2,
            robust_bundle=eq_r3,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="regime_blend",
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa se o universo 800 ajuda a tirar alpha da perna de ações e reduzir dependência do cripto.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10_000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_u800_alpha_suite")
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
    context = built["context"]
    profiles = context["profiles"]

    u800_equities = _build_u800_equity_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        asset_groups=(ROOT / args.equity_asset_groups).resolve(),
        asset_metadata=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_ticker=str(args.benchmark_equity),
        profile=profiles["foreign"],
    )

    crypto_top1 = _crypto_bundle(
        candidate_id="crypto_major8__mom_total_lb021_rb07_k1",
        allowed_tickers=list(context["crypto_tiers"]["crypto_major8"]),
        context=context,
    )
    candidates: dict[str, StrategyBundle] = {
        str(built["attack"].result.candidate_id): built["attack"],
        "alpha_attack_u800_a2r1": _attack_from_parts(
            candidate_id="alpha_attack_u800_a2r1",
            label="ataque com perna de ações do universo 800 em modo trail switch a2r1",
            crypto_bundle=crypto_top1,
            equity_bundle=u800_equities["a2r1"],
            baseline_guard_alloc=built["allocations"]["baseline_guard"],
            context=context,
        ),
        "alpha_attack_u800_a2r3": _attack_from_parts(
            candidate_id="alpha_attack_u800_a2r3",
            label="ataque com perna de ações do universo 800 em modo trail switch a2r3",
            crypto_bundle=crypto_top1,
            equity_bundle=u800_equities["a2r3"],
            baseline_guard_alloc=built["allocations"]["baseline_guard"],
            context=context,
        ),
        "alpha_attack_u800_regime_blend_a2r3": _attack_from_parts(
            candidate_id="alpha_attack_u800_regime_blend_a2r3",
            label="ataque com perna de ações do universo 800 em regime blend a2r3",
            crypto_bundle=crypto_top1,
            equity_bundle=u800_equities["regime_blend_a2r3"],
            baseline_guard_alloc=built["allocations"]["baseline_guard"],
            context=context,
        ),
    }

    top_freq = _selection_frequency_for_crypto_rule(
        allowed_tickers=list(context["crypto_tiers"]["crypto_major8"]),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        benchmark_ticker=str(context["benchmark_crypto"]),
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        score_mode="mom_total",
        asset_ma_days=0,
        market_ma_days=200,
    )
    top3 = top_freq.sort_values(["rebalance_count", "ticker"], ascending=[False, True]).head(3)["ticker"].astype(str).tolist()

    fragility_rows: list[dict[str, Any]] = []
    base_total_map = {cid: _safe_float(bundle.result.net_total_return) for cid, bundle in candidates.items()}
    retained_candidates: dict[str, StrategyBundle] = {}
    for key, equity_bundle in {"alpha_attack_u800_a2r1": u800_equities["a2r1"], "alpha_attack_u800_a2r3": u800_equities["a2r3"], "alpha_attack_u800_regime_blend_a2r3": u800_equities["regime_blend_a2r3"]}.items():
        crypto_drop = _crypto_bundle(
            candidate_id=f"{key}__drop_top3",
            allowed_tickers=[t for t in context["crypto_tiers"]["crypto_major8"] if t not in set(top3)],
            context=context,
        )
        retained = _attack_from_parts(
            candidate_id=f"{key}__drop_top3",
            label="ataque com top3 cripto removidos",
            crypto_bundle=crypto_drop,
            equity_bundle=equity_bundle,
            baseline_guard_alloc=built["allocations"]["baseline_guard"],
            context=context,
        )
        retained_candidates[key] = retained
    # Baseline drop-top3
    baseline_drop_crypto = _crypto_bundle(
        candidate_id="alpha_attack_major8_equity25__drop_top3",
        allowed_tickers=[t for t in context["crypto_tiers"]["crypto_major8"] if t not in set(top3)],
        context=context,
    )
    baseline_drop = _attack_from_parts(
        candidate_id="alpha_attack_major8_equity25__drop_top3",
        label="ataque oficial com top3 cripto removidos",
        crypto_bundle=baseline_drop_crypto,
        equity_bundle=context["equity_attack"],
        baseline_guard_alloc=built["allocations"]["baseline_guard"],
        context=context,
    )
    retained_candidates[str(built["attack"].result.candidate_id)] = baseline_drop

    for cid, bundle in candidates.items():
        dropped = retained_candidates[cid]
        base_total = max(base_total_map[cid], 1e-9)
        fragility_rows.append(
            {
                "candidate_id": cid,
                "drop_top3_total_return": _safe_float(dropped.result.net_total_return),
                "top3_retention": _safe_float(dropped.result.net_total_return) / base_total,
                "drop_top3_ann_return": _safe_float(dropped.result.net_ann_return),
            }
        )

    compare_rows = []
    year_rows: list[dict[str, Any]] = []
    for cid, bundle in candidates.items():
        row = _result_row(bundle.result)
        frag = next((r for r in fragility_rows if r["candidate_id"] == cid), {})
        row.update(frag)
        compare_rows.append(row)
        year_rows.extend(_calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl)))
    compare_df = pd.DataFrame(compare_rows).sort_values(["net_ann_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    fragility_df = pd.DataFrame(fragility_rows).sort_values(["top3_retention", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    year_df = pd.DataFrame(year_rows).sort_values(["year", "candidate_id"], ascending=[True, True]).reset_index(drop=True)

    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)
    fragility_df.to_csv(outdir / "fragility_compare.csv", index=False)
    year_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    best_profit = compare_df.iloc[0].to_dict()
    best_retention = fragility_df.iloc[0].to_dict()
    baseline_row = compare_df[compare_df["candidate_id"].astype(str) == str(built["attack"].result.candidate_id)].head(1)
    baseline_profit = baseline_row.iloc[0].to_dict() if not baseline_row.empty else {}

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "baseline_attack": baseline_profit,
        "best_profit_candidate": best_profit,
        "best_retention_candidate": best_retention,
        "top3_crypto_tickers": top3,
        "insights": [
            f"Melhor candidato de lucro no universo 800: {best_profit.get('candidate_id')}.",
            f"Melhor retenção sem top3 cripto: {best_retention.get('candidate_id')}.",
            "Se o melhor de lucro e o melhor de retenção forem diferentes, o universo 800 ajudou mais a robustez do que o alpha puro.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / 'candidate_compare.csv'),
            "fragility_compare_csv": str(outdir / 'fragility_compare.csv'),
            "yearbook_reais_csv": str(outdir / 'yearbook_reais.csv'),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_u800_alpha_suite.py",
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
            "fragility_compare_csv": str(outdir / "fragility_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
        },
        extra={
            "baseline_ann_return": _safe_float(baseline_profit.get("net_ann_return")),
            "best_profit_ann_return": _safe_float(best_profit.get("net_ann_return")),
            "best_retention": _safe_float(best_retention.get("top3_retention")),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
