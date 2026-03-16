#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import AllocationBundle, _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_champion_timing_robustness_suite import _underperform_prob_rolling  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import build_official_mode_allocations  # noqa: E402
from scripts.bench.validation.run_profit_official_post_fiscal_validation import _topk_crypto_share  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import _build_custom_candidates  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _forward_total_return(net_ret: pd.Series, *, horizon: int) -> pd.Series:
    ret = pd.to_numeric(net_ret, errors="coerce").dropna().astype(float)
    if ret.empty or len(ret) <= int(horizon):
        return pd.Series(dtype=float)
    values = ret.to_numpy(dtype=float)
    out = np.full(ret.shape[0], np.nan, dtype=float)
    hh = int(horizon)
    for i in range(0, len(values) - hh):
        out[i] = float(np.prod(1.0 + values[i + 1 : i + 1 + hh]) - 1.0)
    return pd.Series(out, index=ret.index, dtype=float).dropna()


def _classify_regime_series(structure_daily: pd.DataFrame) -> pd.Series:
    if structure_daily.empty:
        return pd.Series(dtype=str)
    crit = pd.to_numeric(structure_daily.get("criticality"), errors="coerce").fillna(0.5).astype(float)
    stress = pd.to_numeric(structure_daily.get("structural_stress"), errors="coerce").fillna(0.5).astype(float)
    market = pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce")
    if market.isna().all():
        market = pd.to_numeric(structure_daily.get("market_mode_share"), errors="coerce").fillna(0.5).astype(float)
    else:
        market = market.fillna(0.5).astype(float)
    regime = pd.Series("stable", index=structure_daily.index, dtype=object)
    regime[(stress >= 0.70) | (crit >= 0.72) | (market >= 0.82)] = "stress"
    regime[(stress <= 0.42) & (crit <= 0.40) & (market <= 0.35)] = "dispersion"
    regime[
        (regime == "stable")
        & ((stress >= 0.52) | (crit >= 0.55) | (market >= 0.65))
    ] = "transition"
    return regime.astype(str)


def _timing_diagnostics(
    *,
    net_ret: pd.Series,
    benchmark_net_ret: pd.Series,
    structure_daily: pd.DataFrame,
    horizon: int = 21,
) -> tuple[dict[str, float], pd.DataFrame]:
    fwd = _forward_total_return(net_ret, horizon=horizon)
    fwd_bench = _forward_total_return(benchmark_net_ret, horizon=horizon)
    idx = fwd.index.intersection(fwd_bench.index).intersection(structure_daily.index)
    if idx.empty:
        empty = pd.DataFrame(columns=["regime", "n_obs", "avg_alpha_21d", "win_rate_21d"])
        return {
            "regime_separation_21d": float("nan"),
            "criticality_spread_21d": float("nan"),
            "stable_dispersion_alpha_21d": float("nan"),
            "transition_stress_alpha_21d": float("nan"),
        }, empty
    alpha = (pd.to_numeric(fwd.reindex(idx), errors="coerce") - pd.to_numeric(fwd_bench.reindex(idx), errors="coerce")).astype(float)
    regime = _classify_regime_series(structure_daily.reindex(idx))
    crit = pd.to_numeric(structure_daily.get("criticality"), errors="coerce").reindex(idx).fillna(0.5).astype(float)

    rows: list[dict[str, Any]] = []
    for state in ["dispersion", "stable", "transition", "stress"]:
        mask = regime == state
        sub = alpha.loc[mask]
        if sub.empty:
            continue
        rows.append(
            {
                "regime": state,
                "n_obs": int(sub.shape[0]),
                "avg_alpha_21d": float(sub.mean()),
                "win_rate_21d": float((sub > 0.0).mean()),
            }
        )
    regime_df = pd.DataFrame(rows)
    good = alpha.loc[regime.isin(["dispersion", "stable"])]
    bad = alpha.loc[regime.isin(["transition", "stress"])]
    low_crit = alpha.loc[crit <= float(crit.quantile(0.35))]
    high_crit = alpha.loc[crit >= float(crit.quantile(0.65))]
    summary = {
        "regime_separation_21d": float(good.mean() - bad.mean()) if not good.empty and not bad.empty else float("nan"),
        "criticality_spread_21d": float(low_crit.mean() - high_crit.mean()) if not low_crit.empty and not high_crit.empty else float("nan"),
        "stable_dispersion_alpha_21d": float(good.mean()) if not good.empty else float("nan"),
        "transition_stress_alpha_21d": float(bad.mean()) if not bad.empty else float("nan"),
    }
    return summary, regime_df


def _bundle_row(
    *,
    scenario: str,
    sleeve_mode: str,
    observer_mode: str,
    bundle: AllocationBundle,
    crypto_tickers: list[str],
    observer_count: int,
    structure_daily: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result = bundle.bundle.result
    row = {
        "scenario": scenario,
        "sleeve_mode": sleeve_mode,
        "observer_mode": observer_mode,
        "candidate_id": str(result.candidate_id),
        "observer_crypto_count": int(observer_count),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "underperform_prob_63": _underperform_prob_rolling(result.net_ret, result.benchmark_net_ret, horizon=63),
    }
    top1_mean, top1_max = _topk_crypto_share(bundle.weights, crypto_tickers=crypto_tickers, k=1)
    top3_mean, top3_max = _topk_crypto_share(bundle.weights, crypto_tickers=crypto_tickers, k=3)
    row["crypto_top1_share_mean"] = top1_mean
    row["crypto_top1_share_max"] = top1_max
    row["crypto_top3_share_mean"] = top3_mean
    row["crypto_top3_share_max"] = top3_max
    regime_df = pd.DataFrame()
    if structure_daily is not None and not structure_daily.empty:
        diag, regime_df = _timing_diagnostics(
            net_ret=result.net_ret,
            benchmark_net_ret=result.benchmark_net_ret,
            structure_daily=structure_daily,
            horizon=21,
        )
        row.update(diag)
    return row, regime_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara universos cripto de execucao e observacao para decidir promocao ao core.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--current-crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--current-crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--expanded-crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_expanded.csv")
    ap.add_argument("--expanded-crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_expanded.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_crypto_universe_promotion_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()
    current_groups = (ROOT / args.current_crypto_asset_groups).resolve()
    current_meta = (ROOT / args.current_crypto_asset_metadata).resolve()
    expanded_groups = (ROOT / args.expanded_crypto_asset_groups).resolve()
    expanded_meta = (ROOT / args.expanded_crypto_asset_metadata).resolve()
    equity_groups = (ROOT / args.equity_asset_groups).resolve()
    equity_meta = (ROOT / args.equity_asset_metadata).resolve()

    print("[promotion] building current official universe", flush=True)
    official_current = build_official_mode_allocations(
        prices_dir=prices_dir,
        crypto_groups=current_groups,
        crypto_meta=current_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    print("[promotion] building expanded challenger universe", flush=True)
    expanded_built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=expanded_groups,
        crypto_meta=expanded_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    expanded_context = dict(expanded_built["context"])

    print("[promotion] evaluating observer scenarios", flush=True)
    observer_scenarios = [
        (
            "observer_current_major8",
            official_current,
            list(official_current["context"]["crypto_tiers"]["crypto_major8"]),
            "current_major8",
        ),
        (
            "observer_current_all22",
            build_official_mode_allocations(
                prices_dir=prices_dir,
                crypto_groups=current_groups,
                crypto_meta=current_meta,
                equity_groups=equity_groups,
                equity_meta=equity_meta,
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                observer_context=official_current["context"],
                observer_crypto_tickers=list(official_current["context"]["crypto_tiers"]["crypto_all"]),
            ),
            list(official_current["context"]["crypto_tiers"]["crypto_all"]),
            "current_all22",
        ),
        (
            "observer_expanded_major20",
            build_official_mode_allocations(
                prices_dir=prices_dir,
                crypto_groups=current_groups,
                crypto_meta=current_meta,
                equity_groups=equity_groups,
                equity_meta=equity_meta,
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                observer_context=expanded_context,
                observer_crypto_tickers=list(expanded_context["crypto_tiers"]["crypto_major20"]),
            ),
            list(expanded_context["crypto_tiers"]["crypto_major20"]),
            "expanded_major20",
        ),
        (
            "observer_expanded_all",
            build_official_mode_allocations(
                prices_dir=prices_dir,
                crypto_groups=current_groups,
                crypto_meta=current_meta,
                equity_groups=equity_groups,
                equity_meta=equity_meta,
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                observer_context=expanded_context,
                observer_crypto_tickers=list(expanded_context["crypto_tiers"]["crypto_all"]),
            ),
            list(expanded_context["crypto_tiers"]["crypto_all"]),
            "expanded_all40",
        ),
    ]

    observer_rows: list[dict[str, Any]] = []
    observer_regime_rows: list[dict[str, Any]] = []
    for scenario_id, scenario_payload, observer_crypto, observer_mode in observer_scenarios:
        row, regime_df = _bundle_row(
            scenario=scenario_id,
            sleeve_mode="live_major8",
            observer_mode=observer_mode,
            bundle=scenario_payload["official_attack"],
            crypto_tickers=list(official_current["context"]["crypto_tiers"]["crypto_all"]),
            observer_count=len(observer_crypto),
            structure_daily=scenario_payload["structure_daily"],
        )
        observer_rows.append(row)
        if not regime_df.empty:
            frame = regime_df.copy()
            frame.insert(0, "scenario", scenario_id)
            frame.insert(1, "observer_mode", observer_mode)
            observer_regime_rows.extend(frame.to_dict(orient="records"))

    observer_compare = pd.DataFrame(observer_rows).sort_values(
        ["regime_separation_21d", "net_ann_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    observer_compare.to_csv(outdir / "observer_compare.csv", index=False)
    pd.DataFrame(observer_regime_rows).to_csv(outdir / "observer_regime_compare.csv", index=False)

    print("[promotion] evaluating execution scenarios", flush=True)
    execution_specs = [
        ("exec_current_major8", current_groups, current_meta, "major8"),
        ("exec_current_all22", current_groups, current_meta, "all22"),
        ("exec_expanded_major8", expanded_groups, expanded_meta, "major8"),
        ("exec_expanded_all40", expanded_groups, expanded_meta, "all22"),
    ]
    execution_rows: list[dict[str, Any]] = []
    for scenario_id, groups_path, meta_path, crypto_mode in execution_specs:
        payload = _build_custom_candidates(
            prices_dir=prices_dir,
            crypto_groups=groups_path,
            crypto_meta=meta_path,
            equity_groups=equity_groups,
            equity_meta=equity_meta,
            benchmark_crypto=str(args.benchmark_crypto),
            benchmark_equity=str(args.benchmark_equity),
            crypto_allowed_mode=crypto_mode,
        )
        crypto_all = list(payload["crypto_all22"])
        for sleeve_name in ("baseline", "attack", "baseline_guard", "attack_guard"):
            row, _ = _bundle_row(
                scenario=scenario_id,
                sleeve_mode=sleeve_name,
                observer_mode=crypto_mode,
                bundle=payload[sleeve_name],
                crypto_tickers=crypto_all,
                observer_count=len(crypto_all),
                structure_daily=None,
            )
            execution_rows.append(row)

    execution_compare = pd.DataFrame(execution_rows).sort_values(
        ["sleeve_mode", "net_ann_return", "net_sharpe"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    execution_compare.to_csv(outdir / "execution_compare.csv", index=False)

    base_observer = observer_compare.loc[observer_compare["scenario"] == "observer_current_major8"].iloc[0].to_dict()
    best_observer = observer_compare.iloc[0].to_dict() if not observer_compare.empty else {}
    attack_exec = execution_compare.loc[execution_compare["sleeve_mode"] == "attack"].copy()
    attack_exec = attack_exec.sort_values(["net_ann_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    best_attack_exec = attack_exec.iloc[0].to_dict() if not attack_exec.empty else {}

    keep_expanded_observer = bool(
        best_observer
        and str(best_observer.get("observer_mode", "")).startswith("expanded")
        and _safe_float(best_observer.get("regime_separation_21d")) > _safe_float(base_observer.get("regime_separation_21d"))
    )
    promote_expanded_execution = bool(
        best_attack_exec
        and str(best_attack_exec.get("scenario", "")).startswith("exec_expanded")
        and _safe_float(best_attack_exec.get("net_ann_return")) > _safe_float(
            attack_exec.loc[attack_exec["scenario"] == "exec_current_major8", "net_ann_return"].max()
        )
    )

    summary = {
        "suite": "profit_crypto_universe_promotion_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_universe_assets": int(len(official_current["context"]["crypto_tiers"]["crypto_all"])),
        "expanded_universe_assets": int(len(expanded_context["crypto_tiers"]["crypto_all"])),
        "best_observer": best_observer,
        "best_attack_execution": best_attack_exec,
        "keep_expanded_observer": keep_expanded_observer,
        "promote_expanded_execution": promote_expanded_execution,
        "insights": [
            "Observer scenarios mantêm o sleeve live em major8 e trocam apenas o universo usado para diagnosticar criticidade e regime.",
            "Execution scenarios trocam o universo cripto que pode entrar de fato na perna executável.",
            "Se o universo expandido só melhora separação de regime, ele vale como observador; se também melhora net_ann_return e underperform, ele vira candidato de promoção.",
        ],
        "artifacts": {
            "observer_compare_csv": str(outdir / "observer_compare.csv"),
            "observer_regime_compare_csv": str(outdir / "observer_regime_compare.csv"),
            "execution_compare_csv": str(outdir / "execution_compare.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    print("[promotion] writing manifest", flush=True)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_crypto_universe_promotion_suite.py",
        params={
            "prices_dir": str(args.prices_dir),
            "current_crypto_asset_groups": str(args.current_crypto_asset_groups),
            "current_crypto_asset_metadata": str(args.current_crypto_asset_metadata),
            "expanded_crypto_asset_groups": str(args.expanded_crypto_asset_groups),
            "expanded_crypto_asset_metadata": str(args.expanded_crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "observer_compare_csv": str(outdir / "observer_compare.csv"),
            "observer_regime_compare_csv": str(outdir / "observer_regime_compare.csv"),
            "execution_compare_csv": str(outdir / "execution_compare.csv"),
        },
        extra={
            "suite": "profit_crypto_universe_promotion_suite",
            "keep_expanded_observer": keep_expanded_observer,
            "promote_expanded_execution": promote_expanded_execution,
        },
    )
    print("[promotion] done", flush=True)
    print(summary["artifacts"]["observer_compare_csv"])


if __name__ == "__main__":
    main()
