#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import NetAssumptionProfile, apply_net_assumptions  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle, _stress_bundle  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import (  # noqa: E402
    _build_custom_candidates,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _human_label(candidate_id: str) -> str:
    mapping = {
        "alpha_attack_major8_equity25": "Modo ataque",
        "meta_major8_eq_a2r1": "Modo principal",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _stress_profile(profile: NetAssumptionProfile, *, extra_cost_bps: float) -> NetAssumptionProfile:
    return replace(
        profile,
        transaction_cost_bps_assumed=float(profile.transaction_cost_bps_assumed) + float(max(0.0, extra_cost_bps)),
        label=f"{profile.label} + stress",
    )


def _liquidity_proxy_bps(*, capital_brl: float, avg_turnover_daily: float, style: str) -> float:
    base_capital = 10_000.0
    scale = max(0.0, math.log10(max(float(capital_brl), 1.0) / base_capital))
    turnover = max(0.0, float(avg_turnover_daily))
    style_mult = {
        "attack": 1.45,
        "main": 0.95,
        "attack_guard": 1.20,
        "main_guard": 0.80,
    }.get(str(style), 1.0)
    return float(scale * (6.0 + 28.0 * turnover) * style_mult)


def _delay_days_for_style(*, style: str, capital_brl: float) -> int:
    extra = 1 if float(capital_brl) >= 1_000_000.0 else 0
    base = {
        "attack": 1,
        "main": 0,
        "attack_guard": 1,
        "main_guard": 0,
    }.get(str(style), 0)
    return int(base + extra)


def _bundle_with_execution_proxy(
    bundle: StrategyBundle,
    *,
    profile: NetAssumptionProfile,
    capital_brl: float,
    style: str,
) -> dict[str, Any]:
    gross = pd.to_numeric(bundle.result.gross_ret, errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(bundle.result.turnover, errors="coerce").fillna(0.0).astype(float)
    benchmark = pd.to_numeric(bundle.result.benchmark_net_ret, errors="coerce").reindex(gross.index).fillna(0.0).astype(float)

    delay_days = _delay_days_for_style(style=str(style), capital_brl=float(capital_brl))
    if delay_days > 0:
        gross = gross.shift(delay_days).fillna(0.0).astype(float)
        turnover = turnover.shift(delay_days).fillna(0.0).astype(float)

    extra_bps = _liquidity_proxy_bps(
        capital_brl=float(capital_brl),
        avg_turnover_daily=_safe_float(bundle.result.avg_turnover_daily),
        style=str(style),
    )
    stressed_profile = _stress_profile(profile, extra_cost_bps=extra_bps)
    net = apply_net_assumptions(
        gross_ret=gross,
        turnover=turnover,
        profile=stressed_profile,
        periods_index=gross.index,
    )
    net_ret = pd.to_numeric(net["net_ret"], errors="coerce").fillna(0.0).astype(float)
    bench_ret = benchmark

    equity = (1.0 + net_ret).cumprod()
    bench_equity = (1.0 + bench_ret).cumprod()
    ann = float(np.power(float(equity.iloc[-1]), 252.0 / max(int(len(net_ret)), 1)) - 1.0) if len(net_ret) else float("nan")
    vol = float(net_ret.std(ddof=0) * np.sqrt(252.0)) if len(net_ret) else float("nan")
    sharpe = float(ann / vol) if np.isfinite(vol) and vol > 1e-12 else float("nan")
    mdd = float((equity / equity.cummax() - 1.0).min()) if len(net_ret) else float("nan")
    total = float(equity.iloc[-1] - 1.0) if len(net_ret) else float("nan")
    edge = float(total - (bench_equity.iloc[-1] - 1.0)) if len(net_ret) else float("nan")

    return {
        "net_ret": net_ret,
        "benchmark_ret": bench_ret,
        "turnover": turnover,
        "delay_days": int(delay_days),
        "extra_liquidity_bps": float(extra_bps),
        "capital_brl": float(capital_brl),
        "net_ann_return": ann,
        "net_total_return": total,
        "net_sharpe": sharpe,
        "net_max_drawdown": mdd,
        "edge_vs_benchmark": edge,
        "avg_turnover_daily": float(turnover.mean()) if len(turnover) else float("nan"),
        "stressed_profile": stressed_profile,
    }


def _make_yearbook_rows(
    *,
    candidate_id: str,
    candidate_label: str,
    net_ret: pd.Series,
    benchmark_ret: pd.Series,
    turnover: pd.Series,
    capital_brl: float,
    scenario: str,
) -> list[dict[str, Any]]:
    from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult  # noqa: E402

    dummy = StrategyResult(
        suite="capacity_execution",
        candidate_id=str(candidate_id),
        family="capacity_execution",
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=net_ret,
        turnover=turnover,
        net_ret=net_ret,
        benchmark_net_ret=benchmark_ret,
        net_ann_return=float("nan"),
        net_total_return=float("nan"),
        net_sharpe=float("nan"),
        net_max_drawdown=float("nan"),
        edge_vs_benchmark=float("nan"),
        avg_turnover_daily=float("nan"),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=str(scenario),
    )
    rows = _calendar_rows(result=dummy, capital_brl=float(capital_brl))
    for row in rows:
        row["candidate_label"] = str(candidate_label)
        row["scenario"] = str(scenario)
    return rows


def _result_row(candidate_id: str, candidate_label: str, scenario: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "candidate_label": str(candidate_label),
        "scenario": str(scenario),
        "capital_brl": float(payload.get("capital_brl", float("nan"))),
        "delay_days": int(payload.get("delay_days", 0)),
        "extra_liquidity_bps": float(payload.get("extra_liquidity_bps", float("nan"))),
        "net_ann_return": _safe_float(payload.get("net_ann_return")),
        "net_total_return": _safe_float(payload.get("net_total_return")),
        "net_sharpe": _safe_float(payload.get("net_sharpe")),
        "net_max_drawdown": _safe_float(payload.get("net_max_drawdown")),
        "edge_vs_benchmark": _safe_float(payload.get("edge_vs_benchmark")),
        "avg_turnover_daily": _safe_float(payload.get("avg_turnover_daily")),
    }


def _scenario_bundle_row(name: str, allocation_bundle) -> dict[str, Any]:
    result = allocation_bundle.bundle.result
    return {
        "scenario": str(name),
        "candidate_id": str(result.candidate_id),
        "candidate_label": _human_label(result.candidate_id),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Capital crescente, liquidez proxy, atraso e dependencia por grupos.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_capacity_execution_suite")
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

    finalist_bundles = {
        "attack": ("alpha_attack_major8_equity25", "Modo ataque", built["attack"]),
        "main": ("meta_major8_eq_a2r1", "Modo principal", built["baseline"]),
        "attack_guard": ("alpha_attack_major8_equity25_mc_guard", "Modo ataque com guarda", built["attack_guard"]),
        "main_guard": ("meta_major8_eq_a2r1_mc_guard", "Modo principal com guarda", built["baseline_guard"]),
    }

    capital_values = [10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0]
    capital_rows: list[dict[str, Any]] = []
    yearbook_rows: list[dict[str, Any]] = []

    for style, (cid, label, bundle) in finalist_bundles.items():
        for capital in capital_values:
            payload = _bundle_with_execution_proxy(bundle, profile=bundle.profile, capital_brl=capital, style=style)
            scenario = f"capital_{int(capital):d}"
            capital_rows.append(_result_row(cid, label, scenario, payload))
            yearbook_rows.extend(
                _make_yearbook_rows(
                    candidate_id=cid,
                    candidate_label=label,
                    net_ret=payload["net_ret"],
                    benchmark_ret=payload["benchmark_ret"],
                    turnover=payload["turnover"],
                    capital_brl=float(capital),
                    scenario=scenario,
                )
            )

    capital_df = pd.DataFrame(capital_rows).sort_values(["candidate_id", "capital_brl"]).reset_index(drop=True)
    capital_df.to_csv(outdir / "capital_liquidity_delay_compare.csv", index=False)
    pd.DataFrame(yearbook_rows).to_csv(outdir / "yearbook_reais.csv", index=False)

    # Dependencia por grupo.
    base_custom = _build_custom_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    majors = list(base_custom["crypto_major8"])
    scenarios = [
        ("base", {}),
        ("sem_majors", {"crypto_drop_tickers": majors, "crypto_allowed_mode": "all22"}),
        ("sem_setor_technology", {"equity_drop_sectors": ["technology"]}),
        ("sem_dois_grupos_fortes", {"equity_drop_sectors": ["technology", "materials"]}),
    ]
    group_rows: list[dict[str, Any]] = []
    for name, cfg in scenarios:
        if name == "base":
            perturbed = base_custom
        else:
            perturbed = _build_custom_candidates(
                prices_dir=(ROOT / args.prices_dir).resolve(),
                crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
                crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
                equity_groups=(ROOT / args.equity_asset_groups).resolve(),
                equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                crypto_drop_tickers=cfg.get("crypto_drop_tickers"),
                equity_drop_sectors=cfg.get("equity_drop_sectors"),
                crypto_allowed_mode=str(cfg.get("crypto_allowed_mode", "major8")),
            )
        for key in ["attack", "baseline", "attack_guard", "baseline_guard"]:
            group_rows.append(_scenario_bundle_row(name, perturbed[key]))
    group_df = pd.DataFrame(group_rows).sort_values(["candidate_id", "scenario"]).reset_index(drop=True)
    group_df.to_csv(outdir / "group_dependency_compare.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capital_scaling_best": capital_df.sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).head(1).to_dict(orient="records")[0] if not capital_df.empty else {},
        "group_dependency_attack": group_df[group_df["candidate_id"] == "alpha_attack_major8_equity25"].to_dict(orient="records"),
        "insights": [
            "Capital crescente aqui usa um proxy conservador de liquidez, porque a base local não traz volume diário.",
            "Atraso por ativo foi aproximado por atraso maior nos modos mais concentrados em cripto e menor nos mais distribuídos.",
            "Dependência por grupo testa exatamente o que mais importa: sem majors, sem technology e sem os dois grupos fortes juntos.",
        ],
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir / "RUN_MANIFEST.json",
        script=str(Path(__file__).resolve()),
        params={
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "capital_values": capital_values,
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "capital_csv": str(outdir / "capital_liquidity_delay_compare.csv"),
            "group_csv": str(outdir / "group_dependency_compare.csv"),
            "yearbook_csv": str(outdir / "yearbook_reais.csv"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
