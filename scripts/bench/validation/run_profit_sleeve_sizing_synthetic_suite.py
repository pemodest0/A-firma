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

from engine.portfolio import covariance_cholesky  # noqa: E402
from engine.structural import estimate_corr, spectral_pack  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _build_candidates,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import (  # noqa: E402
    _calendar_rows,
)
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    AllocationBundle,
    StrategyBundle,
    _evaluate_allocation_candidate,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import _load_structural_regime_series_local  # noqa: E402
from scripts.lab.run_corr_macro_offline import _classify_regime  # type: ignore # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _human_label(candidate_id: str) -> str:
    mapping = {
        "pure_crypto_attack": "Cripto puro agressivo",
        "pure_crypto_base": "Cripto puro equilibrado",
        "pure_equity_attack": "Ações puras agressivas",
        "pure_equity_base": "Ações puras equilibradas",
        "blend_half_attack": "Mistura fixa meio a meio agressiva",
        "blend_half_base": "Mistura fixa meio a meio equilibrada",
        "meta_major8_eq_a2r1": "Modo principal",
        "alpha_attack_major8_equity25": "Modo ataque",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda",
        "attack_size_soft": "Modo ataque com tamanho suave",
        "attack_size_hard": "Modo ataque com tamanho duro",
        "attack_size_adaptive": "Modo ataque com tamanho adaptativo",
        "attack_size_crypto_cap70": "Modo ataque com teto no cripto",
        "base_size_soft": "Modo principal com tamanho suave",
        "base_size_adaptive": "Modo principal com tamanho adaptativo",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _result_row(bundle: StrategyBundle, *, group: str) -> dict[str, Any]:
    result = bundle.result
    return {
        "group": str(group),
        "candidate_id": str(result.candidate_id),
        "candidate_label": _human_label(result.candidate_id),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "notes": str(result.notes or ""),
    }


def _weights_frame(index: pd.Index, *, crypto: float, equity: float, cash: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crypto": float(crypto),
            "equity": float(equity),
            "cash": float(cash),
        },
        index=index,
        dtype=float,
    )


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
        notes="generated from frozen sleeve returns",
    )


def _scale_weights(
    *,
    base: AllocationBundle,
    returns_frame: pd.DataFrame,
    regime_series: pd.Series,
    candidate_id: str,
    family: str,
    profile,
    regime_map: dict[str, float] | None = None,
    adaptive: bool = False,
    crypto_cap: float | None = None,
) -> StrategyBundle:
    idx = returns_frame.index.sort_values()
    weights = base.weights.reindex(idx).ffill().fillna(0.0).astype(float)
    source = base.source.reindex(idx).ffill().fillna("cash").astype(str)
    reg = regime_series.reindex(idx).ffill().bfill().astype(str).str.lower()

    if adaptive:
        crypto_map = {"stress": 0.10, "transition": 0.45, "stable": 0.85, "dispersion": 1.00}
        equity_map = {"stress": 0.35, "transition": 0.70, "stable": 0.95, "dispersion": 1.00}
        scale = pd.Series(1.0, index=idx, dtype=float)
        crypto_mask = source.str.contains("crypto", case=False, na=False)
        scale.loc[crypto_mask] = reg.loc[crypto_mask].map(crypto_map).fillna(1.0).astype(float)
        scale.loc[~crypto_mask] = reg.loc[~crypto_mask].map(equity_map).fillna(1.0).astype(float)
    else:
        mapping = regime_map or {"stress": 0.35, "transition": 0.65, "stable": 0.85, "dispersion": 1.0}
        scale = reg.map(mapping).fillna(1.0).astype(float)

    out = weights.copy()
    out[["crypto", "equity"]] = out[["crypto", "equity"]].mul(scale, axis=0)
    if crypto_cap is not None:
        out["crypto"] = out["crypto"].clip(upper=float(crypto_cap))
    out["cash"] = 1.0 - out[["crypto", "equity"]].sum(axis=1)
    out["cash"] = out["cash"].clip(lower=0.0, upper=1.0)
    return _bundle_from_sleeves(
        candidate_id=str(candidate_id),
        family=str(family),
        weights=out,
        returns_frame=returns_frame,
        benchmark_ret=base.bundle.benchmark_gross_ret,
        profile=profile,
    )


def _build_block_corr_cov(
    *,
    n_sectors: int,
    assets_per_sector: int,
    within_corr: float,
    cross_corr: float,
    vol: float,
) -> tuple[np.ndarray, list[str]]:
    n_assets = int(n_sectors) * int(assets_per_sector)
    corr = np.full((n_assets, n_assets), float(cross_corr), dtype=float)
    sector_labels: list[str] = []
    for sector in range(int(n_sectors)):
        start = sector * int(assets_per_sector)
        end = start + int(assets_per_sector)
        corr[start:end, start:end] = float(within_corr)
        sector_labels.extend([f"sector_{sector+1}"] * int(assets_per_sector))
    np.fill_diagonal(corr, 1.0)
    sigma = np.full(n_assets, float(vol), dtype=float)
    cov = np.outer(sigma, sigma) * corr
    return cov, sector_labels


def _simulate_synthetic_shift(
    *,
    seed: int = 23,
    n_sectors: int = 4,
    assets_per_sector: int = 10,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    schedule = [
        ("stable", 260, 0.22, 0.04, 0.012),
        ("transition", 120, 0.42, 0.18, 0.017),
        ("stress", 180, 0.72, 0.48, 0.028),
        ("dispersion", 220, 0.16, 0.00, 0.016),
    ]
    rng = np.random.default_rng(int(seed))
    frames: list[pd.DataFrame] = []
    labels: list[str] = []
    dates = pd.bdate_range("2018-01-02", periods=sum(x[1] for x in schedule))
    cursor = 0
    asset_names = [f"A{i+1:02d}" for i in range(int(n_sectors) * int(assets_per_sector))]
    sector_map: dict[str, str] = {}
    for state, length, within_corr, cross_corr, vol in schedule:
        cov, sectors = _build_block_corr_cov(
            n_sectors=int(n_sectors),
            assets_per_sector=int(assets_per_sector),
            within_corr=float(within_corr),
            cross_corr=float(cross_corr),
            vol=float(vol),
        )
        if not sector_map:
            sector_map = {asset: sector for asset, sector in zip(asset_names, sectors)}
        chol = covariance_cholesky(cov)
        z = rng.standard_normal((int(length), len(asset_names)))
        block = z @ chol.T
        if str(state) == "stress":
            block += -0.0008
        elif str(state) == "dispersion":
            base_bias = np.array([0.0010, 0.0002, -0.0001, 0.0006], dtype=float)
            sector_bias = np.repeat(np.resize(base_bias, int(n_sectors)), int(assets_per_sector))
            block += sector_bias
        elif str(state) == "stable":
            block += 0.0003
        frame = pd.DataFrame(block, index=dates[cursor : cursor + int(length)], columns=asset_names)
        frames.append(frame)
        labels.extend([str(state)] * int(length))
        cursor += int(length)
    returns = pd.concat(frames, axis=0).sort_index()
    return returns, pd.Series(labels, index=returns.index, dtype=object), pd.Series(sector_map, dtype=object)


def _synthetic_structural_series(
    returns: pd.DataFrame,
    *,
    window: int = 120,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prev_vec: np.ndarray | None = None
    for end in range(int(window) - 1, len(returns)):
        dt = returns.index[end]
        block = returns.iloc[end - int(window) + 1 : end + 1]
        corr = estimate_corr(block.to_numpy(dtype=float), method="ledoit_wolf")
        vals, vecs = np.linalg.eigh(corr)
        order = np.argsort(vals)[::-1]
        eigvals = vals[order]
        eigvec = np.asarray(vecs[:, order][:, 0], dtype=float)
        if eigvec[np.argmax(np.abs(eigvec))] < 0.0:
            eigvec = -eigvec
        overlap = float("nan")
        if prev_vec is not None and prev_vec.size == eigvec.size:
            overlap = float(abs(np.dot(prev_vec, eigvec) / (np.linalg.norm(prev_vec) * np.linalg.norm(eigvec))))
        prev_vec = eigvec
        pack = spectral_pack(eigvals, topk=min(5, len(eigvals)))
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "p1": _safe_float(pack.get("phi")),
                "deff": _safe_float(pack.get("deff")),
                "eigvec_overlap_1d": overlap,
                "insufficient_universe": False,
                "lambda1": _safe_float(pack.get("lambda1")),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_shift_summary(ts: pd.DataFrame, planted: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    planted_aligned = planted.reindex(pd.to_datetime(ts["date"])).astype(str).str.lower().reset_index(drop=True)
    work = ts.copy()
    work["true_state"] = planted_aligned.to_numpy(dtype=object)
    classified, meta = _classify_regime(
        work,
        hysteresis_days=5,
        exp_stress=0.10,
        exp_transition=0.40,
        exp_stable=0.70,
        exp_dispersion=0.90,
        w_dp1=0.45,
        w_ddeff=0.45,
        w_overlap=0.10,
        threshold_mode="walk_forward",
        walkforward_min_history=120,
    )
    if classified.empty:
        return classified, {"status": "empty"}
    classified["true_state"] = planted.reindex(pd.to_datetime(classified["date"])).astype(str).str.lower().to_numpy(dtype=object)
    grouped = (
        classified.groupby("true_state")[["p1", "deff", "lambda1"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("true_state")
    )
    nonstable_true = classified["true_state"].isin(["transition", "stress"]).astype(int)
    nonstable_pred = classified["regime"].isin(["transition", "stress"]).astype(int)
    exact_match = float((classified["true_state"].astype(str) == classified["regime"].astype(str)).mean())
    change_points = []
    state_series = classified["true_state"].astype(str)
    dates = pd.to_datetime(classified["date"])
    for i in range(1, len(classified)):
        if state_series.iloc[i] != state_series.iloc[i - 1]:
            cp_date = dates.iloc[i]
            target = str(state_series.iloc[i])
            later = classified.iloc[i:]
            hit = later[pd.Series(later["regime"]).astype(str).eq(target)]
            lag_days = float((pd.to_datetime(hit.iloc[0]["date"]) - cp_date).days) if not hit.empty else float("nan")
            change_points.append({"change_date": str(cp_date.date()), "target_state": target, "lag_days": lag_days})
    summary = {
        "status": "ok",
        "exact_match": exact_match,
        "nonstable_recall": float((nonstable_pred[nonstable_true == 1].mean()) if int(nonstable_true.sum()) > 0 else float("nan")),
        "nonstable_precision": float((nonstable_true[nonstable_pred == 1].mean()) if int(nonstable_pred.sum()) > 0 else float("nan")),
        "regime_meta": meta,
        "change_points": change_points,
        "state_means": grouped.to_dict(orient="records"),
    }
    return classified, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Torneio de sleeves, sizing, yearbook e mercado sintético com quebra de correlação.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_sleeve_sizing_synthetic_suite")
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

    attack_alloc = built["allocations"]["attack"]
    base_alloc = built["allocations"]["baseline"]
    attack_sleeves = built["sleeve_returns"]["attack"]
    base_sleeves = built["sleeve_returns"]["baseline"]
    regime_series = _load_structural_regime_series_local(ROOT)

    profile_attack = attack_alloc.bundle.profile
    profile_base = base_alloc.bundle.profile

    tournament: list[StrategyBundle] = [
        _bundle_from_sleeves(
            candidate_id="pure_crypto_attack",
            family="sleeve_tournament",
            weights=_weights_frame(attack_sleeves.index, crypto=1.0, equity=0.0, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=profile_attack,
        ),
        _bundle_from_sleeves(
            candidate_id="pure_crypto_base",
            family="sleeve_tournament",
            weights=_weights_frame(base_sleeves.index, crypto=1.0, equity=0.0, cash=0.0),
            returns_frame=base_sleeves,
            benchmark_ret=base_alloc.bundle.benchmark_gross_ret,
            profile=profile_base,
        ),
        _bundle_from_sleeves(
            candidate_id="pure_equity_attack",
            family="sleeve_tournament",
            weights=_weights_frame(attack_sleeves.index, crypto=0.0, equity=1.0, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=profile_attack,
        ),
        _bundle_from_sleeves(
            candidate_id="pure_equity_base",
            family="sleeve_tournament",
            weights=_weights_frame(base_sleeves.index, crypto=0.0, equity=1.0, cash=0.0),
            returns_frame=base_sleeves,
            benchmark_ret=base_alloc.bundle.benchmark_gross_ret,
            profile=profile_base,
        ),
        _bundle_from_sleeves(
            candidate_id="blend_half_attack",
            family="sleeve_tournament",
            weights=_weights_frame(attack_sleeves.index, crypto=0.5, equity=0.5, cash=0.0),
            returns_frame=attack_sleeves,
            benchmark_ret=attack_alloc.bundle.benchmark_gross_ret,
            profile=profile_attack,
        ),
        _bundle_from_sleeves(
            candidate_id="blend_half_base",
            family="sleeve_tournament",
            weights=_weights_frame(base_sleeves.index, crypto=0.5, equity=0.5, cash=0.0),
            returns_frame=base_sleeves,
            benchmark_ret=base_alloc.bundle.benchmark_gross_ret,
            profile=profile_base,
        ),
        base_alloc.bundle,
        attack_alloc.bundle,
        built["baseline_guard"],
        built["attack_guard"],
    ]
    tournament_df = pd.DataFrame([_result_row(bundle, group="sleeve_tournament") for bundle in tournament]).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    tournament_df.to_csv(outdir / "sleeve_tournament.csv", index=False)

    sizing: list[StrategyBundle] = [
        attack_alloc.bundle,
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_soft",
            family="sizing",
            profile=profile_attack,
            regime_map={"stress": 0.35, "transition": 0.65, "stable": 0.85, "dispersion": 1.0},
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_hard",
            family="sizing",
            profile=profile_attack,
            regime_map={"stress": 0.10, "transition": 0.45, "stable": 0.75, "dispersion": 1.0},
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_adaptive",
            family="sizing",
            profile=profile_attack,
            adaptive=True,
        ),
        _scale_weights(
            base=attack_alloc,
            returns_frame=attack_sleeves,
            regime_series=regime_series,
            candidate_id="attack_size_crypto_cap70",
            family="sizing",
            profile=profile_attack,
            adaptive=True,
            crypto_cap=0.70,
        ),
        _scale_weights(
            base=base_alloc,
            returns_frame=base_sleeves,
            regime_series=regime_series,
            candidate_id="base_size_soft",
            family="sizing",
            profile=profile_base,
            regime_map={"stress": 0.35, "transition": 0.65, "stable": 0.85, "dispersion": 1.0},
        ),
        _scale_weights(
            base=base_alloc,
            returns_frame=base_sleeves,
            regime_series=regime_series,
            candidate_id="base_size_adaptive",
            family="sizing",
            profile=profile_base,
            adaptive=True,
        ),
    ]
    sizing_df = pd.DataFrame([_result_row(bundle, group="sizing") for bundle in sizing]).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    sizing_df.to_csv(outdir / "sizing_compare.csv", index=False)

    finalists: dict[str, StrategyBundle] = {bundle.result.candidate_id: bundle for bundle in tournament + sizing}
    top_ids = pd.concat([tournament_df, sizing_df], axis=0).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    )["candidate_id"].drop_duplicates().head(6).astype(str).tolist()
    yearbook_rows: list[dict[str, Any]] = []
    for candidate_id in top_ids:
        bundle = finalists[str(candidate_id)]
        yearbook_rows.extend(_calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl)))
    yearbook_df = pd.DataFrame(yearbook_rows).sort_values(["year", "profit_brl"], ascending=[True, False]).reset_index(drop=True)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    synthetic_returns, synthetic_states, synthetic_sector_map = _simulate_synthetic_shift()
    synthetic_ts = _synthetic_structural_series(synthetic_returns, window=120)
    synthetic_classified, synthetic_summary = _synthetic_shift_summary(synthetic_ts, synthetic_states)
    synthetic_ts.to_csv(outdir / "synthetic_structural_series.csv", index=False)
    synthetic_classified.to_csv(outdir / "synthetic_regime_compare.csv", index=False)
    (outdir / "synthetic_sector_map.json").write_text(json.dumps(synthetic_sector_map.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    research_rows = []
    top_keep = pd.concat([tournament_df, sizing_df], axis=0).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    ).head(4)
    for _, row in top_keep.iterrows():
        bundle = finalists[str(row["candidate_id"])]
        status = "keep" if str(row["candidate_id"]) in {str(top_ids[0]), str(top_ids[1])} else "watch"
        research_rows.append(
            _research_row(
                bundle.result,
                outdir=outdir,
                status=status,
                methodology="sleeve_tournament_sizing_synthetic",
                label=_human_label(str(row["candidate_id"])),
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sleeve_tournament_winner": tournament_df.iloc[0].to_dict() if not tournament_df.empty else {},
        "sizing_winner": sizing_df.iloc[0].to_dict() if not sizing_df.empty else {},
        "yearbook_top_ids": top_ids,
        "synthetic_summary": synthetic_summary,
        "insights": [
            "O torneio de sleeves mede se o lucro vem do motor inteiro ou se está concentrado em uma perna específica.",
            "O sizing varia o tamanho da aposta por convicção para testar se o ganho aparece mais por dosagem do que por escolha de sleeve.",
            "O yearbook em reais mostra o que seria vendável na prática, com lucro, perda e giro por ano.",
            "O mercado sintético planta uma mudança real de correlação para verificar se a matriz, os autovalores e o classificador reagem.",
        ],
        "artifacts": {
            "sleeve_tournament_csv": str(outdir / "sleeve_tournament.csv"),
            "sizing_compare_csv": str(outdir / "sizing_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "synthetic_structural_series_csv": str(outdir / "synthetic_structural_series.csv"),
            "synthetic_regime_compare_csv": str(outdir / "synthetic_regime_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_sleeve_sizing_synthetic_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "capital_brl": args.capital_brl,
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
