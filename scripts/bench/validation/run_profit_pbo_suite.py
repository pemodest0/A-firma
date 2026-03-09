#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
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
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import _load_structural_regime_series_local  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_sleeve_sizing_synthetic_suite import (  # noqa: E402
    _bundle_from_sleeves,
    _scale_weights,
    _weights_frame,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _human_label(candidate_id: str) -> str:
    mapping = {
        "meta_major8_eq_a2r1": "Modo principal",
        "alpha_attack_major8_equity25": "Modo ataque",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda",
        "pure_crypto_attack": "Cripto puro agressivo",
        "pure_equity_attack": "Ações puras agressivas",
        "blend_half_attack": "Mistura fixa meio a meio",
        "attack_size_soft": "Ataque com tamanho suave",
        "attack_size_hard": "Ataque com tamanho duro",
        "attack_size_adaptive": "Ataque com tamanho adaptativo",
        "attack_size_crypto_cap70": "Ataque com teto no cripto",
        "base_size_soft": "Principal com tamanho suave",
        "base_size_adaptive": "Principal com tamanho adaptativo",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _monthly_returns(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return pd.Series(dtype=float)
    monthly = x.groupby(pd.to_datetime(x.index).to_period("M")).apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
    monthly.index = monthly.index.astype(str)
    return monthly.astype(float)


def _ann_sharpe_monthly(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if x.shape[0] < 3:
        return float("nan")
    vol = float(x.std(ddof=0))
    if vol <= 1e-12:
        return float("nan")
    return float(x.mean() / vol * math.sqrt(12.0))


def _total_return(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)


def _metric_value(returns: pd.Series, metric: str) -> float:
    if str(metric) == "sharpe":
        return _ann_sharpe_monthly(returns)
    if str(metric) == "total_return":
        return _total_return(returns)
    raise ValueError(f"unsupported metric: {metric}")


def _build_candidate_bundles(args: argparse.Namespace) -> list[Any]:
    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
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
    base_sleeves = built["sleeve_returns"]["baseline"]
    regime_series = _load_structural_regime_series_local(ROOT)

    bundles = [
        _bundle_from_sleeves(
            candidate_id="pure_crypto_attack",
            family="pbo_family",
            weights=_weights_frame(attack_sleeves.index, crypto=1.0, equity=0.0, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=attack_alloc.bundle.profile,
        ),
        _bundle_from_sleeves(
            candidate_id="pure_equity_attack",
            family="pbo_family",
            weights=_weights_frame(attack_sleeves.index, crypto=0.0, equity=1.0, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=attack_alloc.bundle.profile,
        ),
        _bundle_from_sleeves(
            candidate_id="blend_half_attack",
            family="pbo_family",
            weights=_weights_frame(attack_sleeves.index, crypto=0.5, equity=0.5, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=attack_alloc.bundle.profile,
        ),
        built["baseline"],
        built["attack"],
        built["baseline_guard"],
        built["attack_guard"],
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_soft",
            family="pbo_family",
            profile=attack_alloc.bundle.profile,
            regime_map={"stress": 0.35, "transition": 0.65, "stable": 0.85, "dispersion": 1.0},
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_hard",
            family="pbo_family",
            profile=attack_alloc.bundle.profile,
            regime_map={"stress": 0.10, "transition": 0.45, "stable": 0.75, "dispersion": 1.0},
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_adaptive",
            family="pbo_family",
            profile=attack_alloc.bundle.profile,
            adaptive=True,
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_crypto_cap70",
            family="pbo_family",
            profile=attack_alloc.bundle.profile,
            adaptive=True,
            crypto_cap=0.70,
        ),
        _scale_weights(
            base=base_alloc,
            returns_frame=base_sleeves,
            regime_series=regime_series,
            candidate_id="base_size_soft",
            family="pbo_family",
            profile=base_alloc.bundle.profile,
            regime_map={"stress": 0.35, "transition": 0.65, "stable": 0.85, "dispersion": 1.0},
        ),
        _scale_weights(
            base=base_alloc,
            returns_frame=base_sleeves,
            regime_series=regime_series,
            candidate_id="base_size_adaptive",
            family="pbo_family",
            profile=base_alloc.bundle.profile,
            adaptive=True,
        ),
    ]
    return bundles


def _common_monthly_matrix(bundles: list[Any]) -> pd.DataFrame:
    monthly_map: dict[str, pd.Series] = {}
    for bundle in bundles:
        monthly_map[str(bundle.result.candidate_id)] = _monthly_returns(bundle.result.net_ret)
    common = None
    for series in monthly_map.values():
        idx = pd.Index(series.index)
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        return pd.DataFrame()
    common = pd.Index(sorted(common.astype(str).tolist()))
    data = {cid: series.reindex(common).astype(float) for cid, series in monthly_map.items()}
    return pd.DataFrame(data, index=common).dropna(how="any")


def _cscv_splits(index: pd.Index, n_slices: int) -> list[tuple[list[int], list[int]]]:
    slices = [list(x) for x in np.array_split(np.arange(len(index)), int(n_slices)) if len(x) > 0]
    n = len(slices)
    if n < 2 or n % 2 != 0:
        raise ValueError("n_slices must produce an even number of non-empty slices")
    half = n // 2
    combos = list(itertools.combinations(range(n), half))
    unique: list[tuple[list[int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for combo in combos:
        key = tuple(sorted(combo))
        comp = tuple(sorted(set(range(n)) - set(combo)))
        canon = min(key, comp)
        if canon in seen:
            continue
        seen.add(canon)
        unique.append((list(key), list(comp)))
    return unique


def _slice_positions(slices: list[list[int]], chosen: list[int]) -> list[int]:
    pos: list[int] = []
    for idx in chosen:
        pos.extend(slices[int(idx)])
    return sorted(pos)


def _rank_to_omega(rank_desc: int, n_candidates: int) -> float:
    return float((int(n_candidates) - int(rank_desc) + 1) / (int(n_candidates) + 1))


def _pbo_for_metric(matrix: pd.DataFrame, metric: str, n_slices: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    slices = [list(x) for x in np.array_split(np.arange(len(matrix.index)), int(n_slices)) if len(x) > 0]
    splits = _cscv_splits(matrix.index, n_slices=int(n_slices))
    rows: list[dict[str, Any]] = []
    winner_counts: dict[str, int] = {}
    for split_id, (ins, oos) in enumerate(splits, start=1):
        ins_pos = _slice_positions(slices, ins)
        oos_pos = _slice_positions(slices, oos)
        ins_frame = matrix.iloc[ins_pos]
        oos_frame = matrix.iloc[oos_pos]
        ins_scores = {cid: _metric_value(ins_frame[cid], metric) for cid in matrix.columns}
        oos_scores = {cid: _metric_value(oos_frame[cid], metric) for cid in matrix.columns}
        ins_ranked = sorted(ins_scores.items(), key=lambda kv: (-np.nan_to_num(kv[1], nan=-1e18), kv[0]))
        winner = str(ins_ranked[0][0])
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        oos_ranked = sorted(oos_scores.items(), key=lambda kv: (-np.nan_to_num(kv[1], nan=-1e18), kv[0]))
        rank_map = {cid: rank for rank, (cid, _) in enumerate(oos_ranked, start=1)}
        oos_rank = int(rank_map[winner])
        omega = _rank_to_omega(oos_rank, len(matrix.columns))
        lam = float(math.log(omega / (1.0 - omega)))
        rows.append(
            {
                "metric": str(metric),
                "split_id": int(split_id),
                "winner_candidate_id": winner,
                "winner_label": _human_label(winner),
                "winner_in_sample_score": float(ins_scores[winner]),
                "winner_out_of_sample_score": float(oos_scores[winner]),
                "winner_oos_rank_desc": int(oos_rank),
                "winner_oos_omega": omega,
                "winner_oos_logit": lam,
                "winner_below_median": int(omega < 0.5),
            }
        )
    split_df = pd.DataFrame(rows)
    if split_df.empty:
        return split_df, {"status": "empty", "metric": str(metric)}
    summary = {
        "status": "ok",
        "metric": str(metric),
        "n_candidates": int(matrix.shape[1]),
        "n_splits": int(split_df.shape[0]),
        "pbo": float((split_df["winner_below_median"] == 1).mean()),
        "median_omega": float(split_df["winner_oos_omega"].median()),
        "median_logit": float(split_df["winner_oos_logit"].median()),
        "mean_oos_rank_desc": float(split_df["winner_oos_rank_desc"].mean()),
        "winner_counts": winner_counts,
    }
    return split_df, summary


def _pbo_verdict(pbo: float) -> str:
    if not np.isfinite(float(pbo)):
        return "indeterminado"
    if float(pbo) < 0.10:
        return "robusto"
    if float(pbo) < 0.20:
        return "aceitavel"
    if float(pbo) < 0.40:
        return "fragil"
    return "provavel_overfit"


def main() -> None:
    ap = argparse.ArgumentParser(description="Probability of Backtest Overfitting para os modos finais do motor.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--outdir-root", default="results/validation/profit_pbo_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bundles = _build_candidate_bundles(args)
    monthly_matrix = _common_monthly_matrix(bundles)
    if monthly_matrix.empty:
        raise SystemExit("no common monthly matrix for PBO")
    if int(args.n_slices) < 4 or int(args.n_slices) % 2 != 0:
        raise SystemExit("n-slices must be even and >= 4")
    if len(monthly_matrix.index) < int(args.n_slices) * 4:
        raise SystemExit("not enough monthly observations for requested n-slices")

    split_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for metric in ["sharpe", "total_return"]:
        split_df, metric_summary = _pbo_for_metric(monthly_matrix, metric=metric, n_slices=int(args.n_slices))
        split_rows.append(split_df)
        metric_summary["verdict"] = _pbo_verdict(float(metric_summary.get("pbo", float("nan"))))
        metric_rows.append(metric_summary)

    split_df = pd.concat(split_rows, ignore_index=True).sort_values(["metric", "split_id"]).reset_index(drop=True)
    metric_df = pd.DataFrame(metric_rows).sort_values("metric").reset_index(drop=True)
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate_id": str(bundle.result.candidate_id),
                "candidate_label": _human_label(bundle.result.candidate_id),
                "net_ann_return": float(bundle.result.net_ann_return),
                "net_total_return": float(bundle.result.net_total_return),
                "net_sharpe": float(bundle.result.net_sharpe),
                "net_max_drawdown": float(bundle.result.net_max_drawdown),
            }
            for bundle in bundles
        ]
    ).sort_values(["net_total_return", "net_ann_return"], ascending=[False, False]).reset_index(drop=True)

    split_df.to_csv(outdir / "pbo_split_results.csv", index=False)
    metric_df.to_csv(outdir / "pbo_metric_summary.csv", index=False)
    candidate_summary.to_csv(outdir / "candidate_family_summary.csv", index=False)
    monthly_matrix.reset_index(names="ym").to_csv(outdir / "candidate_monthly_matrix.csv", index=False)

    sharpe_row = metric_df[metric_df["metric"] == "sharpe"].head(1)
    total_row = metric_df[metric_df["metric"] == "total_return"].head(1)
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_size": int(candidate_summary.shape[0]),
        "months_common": int(monthly_matrix.shape[0]),
        "primary_metric": "sharpe",
        "pbo_sharpe": sharpe_row.iloc[0].to_dict() if not sharpe_row.empty else {},
        "pbo_total_return": total_row.iloc[0].to_dict() if not total_row.empty else {},
        "overall_verdict": str(sharpe_row.iloc[0]["verdict"]) if not sharpe_row.empty else "indeterminado",
        "insights": [
            "PBO baixo indica que o melhor candidato em treino tende a continuar razoavel no teste.",
            "PBO alto indica que o vencedor do backtest costuma cair para a metade ruim quando sai do treino.",
            "A leitura principal aqui usa Sharpe mensal em blocos simetricos; retorno total entra como criterio secundario.",
        ],
        "artifacts": {
            "pbo_split_results_csv": str(outdir / "pbo_split_results.csv"),
            "pbo_metric_summary_csv": str(outdir / "pbo_metric_summary.csv"),
            "candidate_family_summary_csv": str(outdir / "candidate_family_summary.csv"),
            "candidate_monthly_matrix_csv": str(outdir / "candidate_monthly_matrix.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    research_rows = []
    for bundle in bundles:
        status = "watch"
        cid = str(bundle.result.candidate_id)
        if cid in {"meta_major8_eq_a2r1", "alpha_attack_major8_equity25"}:
            status = "keep"
        research_rows.append(
            _research_row(
                bundle.result,
                outdir=outdir,
                status=status,
                methodology="pbo_audit",
                label=f"{_human_label(cid)} | auditoria PBO",
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_pbo_suite.py",
        params={"n_slices": args.n_slices, "benchmark_crypto": args.benchmark_crypto, "benchmark_equity": args.benchmark_equity},
        paths=summary["artifacts"] | {"profit_research_rows_json": str(outdir / "profit_research_rows.json")},
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
