from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio import (
    build_hmm_feature_frame,
    estimate_regime_moments,
    estimate_transition_matrix,
    fit_hmm_challenger,
    hrp_weights,
    simulate_regime_conditioned_paths,
    summarize_portfolio_distribution,
)
from engine.structural import spectral_pack
from engine.structural.covariance_estimators import estimate_corr
from execution.returns import load_return_series_csv

RAW_RETURNS_DIR = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
CORE_UNIVERSE_PATH = ROOT / "results" / "lab_corr_macro" / "20260223T202847Z" / "universe_core.csv"
REGIME_SERIES_PATH = ROOT / "results" / "lab_corr_macro" / "20260223T202847Z" / "regime_series_T120.csv"
TARGET800_METADATA_PATH = ROOT / "data" / "asset_metadata_target_800_clean_plus.csv"
TARGET800_SUMMARY_PATH = ROOT / "results" / "validation" / "universe_expansion_pack_compare" / "final_structural_verdict.json"
OUTROOT = ROOT / "results" / "validation" / "corr_eigen_sector_analysis"


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _timestamp() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")


def _rank01(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return x.rank(pct=True, method="average").astype(float)


def _orient_eigenvector(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float).copy()
    if np.nansum(arr) < 0:
        arr *= -1.0
    return arr


def _latest_common_window(frame: pd.DataFrame, window: int, min_coverage: float) -> tuple[pd.DataFrame, pd.Series]:
    recent = frame.sort_index().tail(int(window) * 3).copy()
    if recent.empty:
        return recent, pd.Series(dtype=float)
    coverage = recent.tail(int(window)).notna().mean(axis=0)
    keep = coverage[coverage >= float(min_coverage)].index.tolist()
    if not keep:
        return pd.DataFrame(index=recent.index), coverage
    trimmed = recent[keep].tail(int(window)).dropna(axis=0, how="any")
    return trimmed, coverage.loc[keep].sort_values(ascending=False)


def _load_universe_matrix(
    tickers: list[str],
    *,
    window: int,
    min_coverage: float,
    cache: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        if ticker not in cache:
            path = RAW_RETURNS_DIR / f"{ticker}.csv"
            if not path.exists():
                continue
            try:
                cache[ticker] = load_return_series_csv(
                    path,
                    source_kind="log",
                    target_kind="simple",
                    business_days_only=True,
                    series_name=ticker,
                )
            except Exception:
                continue
        if ticker in cache:
            series_map[ticker] = cache[ticker]
    if not series_map:
        return pd.DataFrame(), []
    frame = pd.concat(series_map.values(), axis=1, join="outer", sort=False).sort_index()
    recent, coverage = _latest_common_window(frame, window=window, min_coverage=min_coverage)
    keep = [str(c) for c in coverage.index if c in recent.columns]
    return recent[keep].copy(), keep


def _eig_decompose(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(np.asarray(corr, dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _top_pairwise(corr: pd.DataFrame, *, topn: int = 15) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cols = list(corr.columns)
    mat = corr.to_numpy(dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = _safe_float(mat[i, j])
            if np.isfinite(val):
                rows.append(
                    {
                        "left": cols[i],
                        "right": cols[j],
                        "corr": val,
                        "abs_corr": abs(val),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_corr", "corr"], ascending=[False, False]).head(int(topn)).reset_index(drop=True)


def _compute_detail_tables(
    returns: pd.DataFrame,
    sector_map: pd.Series,
    regime_map: pd.Series,
    *,
    window_label: str,
    covariance_method: str = "ledoit_wolf",
) -> dict[str, object]:
    aligned_sector = sector_map.reindex(returns.columns).fillna("unknown").astype(str)
    corr = pd.DataFrame(
        estimate_corr(returns.to_numpy(dtype=float), method=covariance_method),
        index=returns.columns,
        columns=returns.columns,
    )
    eigvals, eigvecs = _eig_decompose(corr.to_numpy(dtype=float))
    v1 = _orient_eigenvector(eigvecs[:, 0])
    v2 = _orient_eigenvector(eigvecs[:, 1]) if eigvecs.shape[1] > 1 else np.full_like(v1, np.nan)
    pack = spectral_pack(eigvals, topk=min(10, len(eigvals)))
    avg_abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean(axis=1)

    asset_detail = pd.DataFrame(
        {
            "asset": returns.columns.astype(str),
            "sector": aligned_sector.values,
            "v1_loading": v1,
            "v1_abs": np.abs(v1),
            "v2_loading": v2,
            "avg_abs_corr": pd.to_numeric(avg_abs_corr.reindex(returns.columns), errors="coerce").to_numpy(dtype=float),
            "vol_20d": returns.std(ddof=0).reindex(returns.columns).to_numpy(dtype=float),
        }
    )
    asset_detail["systemic_role_score"] = (
        0.6 * _rank01(asset_detail["v1_abs"]).fillna(0.0) + 0.4 * _rank01(asset_detail["avg_abs_corr"]).fillna(0.0)
    )
    asset_detail = asset_detail.sort_values(["systemic_role_score", "v1_abs"], ascending=[False, False]).reset_index(drop=True)

    sector_rows: list[dict[str, object]] = []
    for sector, tickers in asset_detail.groupby("sector")["asset"]:
        names = tickers.tolist()
        if not names:
            continue
        inside = corr.loc[names, names].to_numpy(dtype=float)
        mask_inside = ~np.eye(len(names), dtype=bool)
        outside_names = [c for c in returns.columns if c not in names]
        within_corr_mean = float(np.nanmean(inside[mask_inside])) if mask_inside.any() else float("nan")
        cross_corr_mean = (
            float(corr.loc[names, outside_names].to_numpy(dtype=float).mean()) if outside_names else float("nan")
        )
        sector_rows.append(
            {
                "sector": sector,
                "n_assets": int(len(names)),
                "v1_abs_share": float(asset_detail.loc[asset_detail["sector"] == sector, "v1_abs"].sum() / max(asset_detail["v1_abs"].sum(), 1e-12)),
                "v1_mean_signed": float(asset_detail.loc[asset_detail["sector"] == sector, "v1_loading"].mean()),
                "v1_mean_abs": float(asset_detail.loc[asset_detail["sector"] == sector, "v1_abs"].mean()),
                "within_corr_mean": within_corr_mean,
                "cross_corr_mean": cross_corr_mean,
                "cohesion_gap": float(within_corr_mean - cross_corr_mean) if np.isfinite(within_corr_mean) and np.isfinite(cross_corr_mean) else float("nan"),
                "avg_systemic_role_score": float(asset_detail.loc[asset_detail["sector"] == sector, "systemic_role_score"].mean()),
            }
        )
    sector_detail = pd.DataFrame(sector_rows).sort_values(["v1_abs_share", "cohesion_gap"], ascending=[False, False]).reset_index(drop=True)

    sector_returns = returns.T.groupby(aligned_sector).mean().T
    sector_returns = sector_returns.loc[:, sector_returns.notna().sum(axis=0) >= max(40, int(len(sector_returns) * 0.75))]
    sector_corr = pd.DataFrame(
        estimate_corr(sector_returns.to_numpy(dtype=float), method=covariance_method),
        index=sector_returns.columns,
        columns=sector_returns.columns,
    )
    sector_eigvals, sector_eigvecs = _eig_decompose(sector_corr.to_numpy(dtype=float))
    sector_v1 = _orient_eigenvector(sector_eigvecs[:, 0])
    sector_pack = spectral_pack(sector_eigvals, topk=min(10, len(sector_eigvals)))
    sector_spectrum = pd.DataFrame(
        {
            "sector": sector_returns.columns.astype(str),
            "sector_v1_loading": sector_v1,
            "sector_v1_abs": np.abs(sector_v1),
            "sector_avg_abs_corr": sector_corr.abs().where(~np.eye(len(sector_corr), dtype=bool)).mean(axis=1).to_numpy(dtype=float),
        }
    ).sort_values("sector_v1_abs", ascending=False)

    aligned_regime = regime_map.reindex(sector_returns.index).ffill().bfill()
    valid_mask = aligned_regime.notna()
    sector_returns_reg = sector_returns.loc[valid_mask]
    aligned_regime = aligned_regime.loc[valid_mask].astype(str)
    moments = estimate_regime_moments(sector_returns_reg, aligned_regime, min_obs=20)
    states, transition = estimate_transition_matrix(aligned_regime)
    current_regime = str(aligned_regime.iloc[-1]) if not aligned_regime.empty else "stable"
    sim_paths, sim_states = simulate_regime_conditioned_paths(
        regime_moments=moments,
        transition_matrix=transition,
        states=states,
        start_state=current_regime,
        horizon=21,
        n_paths=1500,
        random_state=23,
    )
    scenario_rows: list[dict[str, object]] = []
    for idx, sector in enumerate(sector_returns.columns):
        weights = np.zeros(len(sector_returns.columns), dtype=float)
        weights[idx] = 1.0
        dist = summarize_portfolio_distribution(sim_paths, weights=weights)
        scenario_rows.append(
            {
                "sector": str(sector),
                "current_regime": current_regime,
                "sim_median_21d": float(dist["terminal_p50"]),
                "sim_p05_21d": float(dist["terminal_p05"]),
                "sim_p95_21d": float(dist["terminal_p95"]),
                "ruin_prob_m10": float(dist["ruin_prob_m10"]),
                "expected_shortfall_21d": float(dist["expected_shortfall_p05"]),
            }
        )
    sector_scenario = pd.DataFrame(scenario_rows)

    sector_cov = sector_returns.cov().astype(float)
    hrp = hrp_weights(sector_cov, corr=sector_corr)
    hrp_df = hrp.rename("hrp_weight").reset_index().rename(columns={"index": "sector"})

    hmm_features = build_hmm_feature_frame(
        primary_ret=sector_returns.mean(axis=1),
        secondary_ret=sector_returns_reg.mean(axis=1).reindex(sector_returns.index).ffill().bfill(),
        volatility_window=21,
    )
    hmm = fit_hmm_challenger(
        hmm_features,
        n_states=3,
        train_end=hmm_features.index[int(len(hmm_features) * 0.7)],
        random_state=23,
    )
    structural_map = aligned_regime.map(
        {
            "stable": "risk_on",
            "dispersion": "risk_on",
            "transition": "neutral",
            "stress": "risk_off",
        }
    )
    hmm_aligned = hmm.regime_label.reindex(structural_map.index).dropna()
    structural_aligned = structural_map.reindex(hmm_aligned.index).astype(str)
    hmm_agreement = float((hmm_aligned == structural_aligned).mean()) if not hmm_aligned.empty else float("nan")

    sector_detail = sector_detail.merge(sector_scenario, on="sector", how="left")
    sector_detail["heuristic_pressure_score"] = (
        0.4 * _rank01(sector_detail["v1_abs_share"]).fillna(0.0)
        + 0.35 * _rank01(sector_detail["cohesion_gap"]).fillna(0.0)
        + 0.25 * _rank01(sector_detail["ruin_prob_m10"]).fillna(0.0)
    )
    sector_detail = sector_detail.sort_values("heuristic_pressure_score", ascending=False).reset_index(drop=True)

    top_assets = asset_detail.head(15)[["asset", "sector", "v1_abs", "avg_abs_corr", "systemic_role_score"]]
    diversifiers = asset_detail.sort_values(["v1_abs", "avg_abs_corr"], ascending=[True, True]).head(15)[
        ["asset", "sector", "v1_abs", "avg_abs_corr", "systemic_role_score"]
    ]

    summary = {
        "window_label": window_label,
        "n_assets": int(returns.shape[1]),
        "n_days": int(returns.shape[0]),
        "asset_pack": {k: _safe_float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in pack.items()},
        "sector_pack": {k: _safe_float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in sector_pack.items()},
        "avg_abs_corr_assets": _safe_float(corr.abs().where(~np.eye(len(corr), dtype=bool)).stack().mean()),
        "avg_abs_corr_sectors": _safe_float(sector_corr.abs().where(~np.eye(len(sector_corr), dtype=bool)).stack().mean()),
        "top_sector_pressure": sector_detail.head(5).to_dict(orient="records"),
        "top_assets_systemic": top_assets.to_dict(orient="records"),
        "top_assets_diversifiers": diversifiers.to_dict(orient="records"),
        "hmm_current_label": str(hmm.regime_label.iloc[-1]) if not hmm.regime_label.empty else "unknown",
        "hmm_risk_on_probability": _safe_float(hmm.risk_on_probability.iloc[-1]) if not hmm.risk_on_probability.empty else float("nan"),
        "hmm_structural_agreement_63d": float(
            (hmm_aligned.tail(63) == structural_aligned.tail(63)).mean()
        )
        if len(hmm_aligned) >= 10
        else float("nan"),
    }
    return {
        "summary": summary,
        "asset_detail": asset_detail,
        "sector_detail": sector_detail,
        "pairwise_top": _top_pairwise(corr, topn=20),
        "sector_pairwise_top": _top_pairwise(sector_corr, topn=20),
        "sector_hrp": hrp_df.sort_values("hrp_weight", ascending=False).reset_index(drop=True),
        "sector_spectrum": sector_spectrum.reset_index(drop=True),
        "hmm_state_summary": hmm.state_summary.copy(),
        "sim_state_path_counts": pd.Series(np.asarray(sim_states, dtype=object).ravel())
        .value_counts(normalize=True)
        .rename("probability")
        .reset_index()
        .rename(columns={"index": "regime"}),
    }


def main() -> None:
    outdir = OUTROOT / _timestamp()
    outdir.mkdir(parents=True, exist_ok=True)

    regime_df = pd.read_csv(REGIME_SERIES_PATH)
    regime_df["date"] = pd.to_datetime(regime_df["date"], errors="coerce")
    regime_map = regime_df.set_index("date")["regime"].astype(str).sort_index()

    core_universe = pd.read_csv(CORE_UNIVERSE_PATH)
    core_sector_map = core_universe.set_index("ticker")["sector"].astype(str)
    clean_meta = pd.read_csv(TARGET800_METADATA_PATH)
    clean_sector_map = clean_meta.set_index("ticker")["sector_internal"].astype(str)

    cache: dict[str, pd.Series] = {}
    core_returns, core_keep = _load_universe_matrix(core_universe["ticker"].astype(str).tolist(), window=120, min_coverage=0.92, cache=cache)
    clean_returns, clean_keep = _load_universe_matrix(clean_meta["ticker"].astype(str).tolist(), window=120, min_coverage=0.90, cache=cache)

    core_bundle = _compute_detail_tables(core_returns, core_sector_map.reindex(core_keep), regime_map, window_label="core_T120")
    clean_bundle = _compute_detail_tables(clean_returns, clean_sector_map.reindex(clean_keep), regime_map, window_label="target800_clean_plus_T120")

    universe_compare = pd.DataFrame(
        [
            {
                "universe": "core_T120",
                **core_bundle["summary"]["asset_pack"],
                "n_assets": core_bundle["summary"]["n_assets"],
                "n_days": core_bundle["summary"]["n_days"],
                "avg_abs_corr_assets": core_bundle["summary"]["avg_abs_corr_assets"],
                "avg_abs_corr_sectors": core_bundle["summary"]["avg_abs_corr_sectors"],
                "hmm_current_label": core_bundle["summary"]["hmm_current_label"],
                "hmm_risk_on_probability": core_bundle["summary"]["hmm_risk_on_probability"],
                "hmm_structural_agreement_63d": core_bundle["summary"]["hmm_structural_agreement_63d"],
            },
            {
                "universe": "target800_clean_plus_T120",
                **clean_bundle["summary"]["asset_pack"],
                "n_assets": clean_bundle["summary"]["n_assets"],
                "n_days": clean_bundle["summary"]["n_days"],
                "avg_abs_corr_assets": clean_bundle["summary"]["avg_abs_corr_assets"],
                "avg_abs_corr_sectors": clean_bundle["summary"]["avg_abs_corr_sectors"],
                "hmm_current_label": clean_bundle["summary"]["hmm_current_label"],
                "hmm_risk_on_probability": clean_bundle["summary"]["hmm_risk_on_probability"],
                "hmm_structural_agreement_63d": clean_bundle["summary"]["hmm_structural_agreement_63d"],
            },
        ]
    )

    for prefix, bundle in [("core", core_bundle), ("target800", clean_bundle)]:
        bundle["asset_detail"].to_csv(outdir / f"{prefix}_asset_detail.csv", index=False)
        bundle["sector_detail"].to_csv(outdir / f"{prefix}_sector_detail.csv", index=False)
        bundle["pairwise_top"].to_csv(outdir / f"{prefix}_pairwise_top.csv", index=False)
        bundle["sector_pairwise_top"].to_csv(outdir / f"{prefix}_sector_pairwise_top.csv", index=False)
        bundle["sector_hrp"].to_csv(outdir / f"{prefix}_sector_hrp.csv", index=False)
        bundle["sector_spectrum"].to_csv(outdir / f"{prefix}_sector_spectrum.csv", index=False)
        bundle["hmm_state_summary"].to_csv(outdir / f"{prefix}_hmm_state_summary.csv", index=False)
        bundle["sim_state_path_counts"].to_csv(outdir / f"{prefix}_sim_state_path_counts.csv", index=False)

    universe_compare.to_csv(outdir / "universe_compare.csv", index=False)

    target800_structural = {}
    if TARGET800_SUMMARY_PATH.exists():
        target800_structural = json.loads(TARGET800_SUMMARY_PATH.read_text())

    summary = {
        "status": "ok",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "outdir": str(outdir),
        "notes": [
            "Matriz estimada com ledoit_wolf em retornos simples dos ultimos 120 dias uteis.",
            "Autovalores e autovetores usados para medir modo dominante, diversificacao efetiva e pressao por setor.",
            "Stress setorial calculado com Monte Carlo condicionado por regime estrutural e correlacoes por Cholesky.",
            "HRP e HMM entram como camadas auxiliares, nao como substitutos do Eigen Engine.",
        ],
        "core_summary": core_bundle["summary"],
        "target800_clean_plus_summary": clean_bundle["summary"],
        "target800_original_structural_context": target800_structural,
        "findings": [
            f"Core T120: p1={core_bundle['summary']['asset_pack'].get('phi'):.4f}, deff={core_bundle['summary']['asset_pack'].get('deff'):.2f}, n_assets={core_bundle['summary']['n_assets']}.",
            f"Target800 clean plus T120: p1={clean_bundle['summary']['asset_pack'].get('phi'):.4f}, deff={clean_bundle['summary']['asset_pack'].get('deff'):.2f}, n_assets={clean_bundle['summary']['n_assets']}.",
            f"Maior pressao setorial no core: {core_bundle['summary']['top_sector_pressure'][0]['sector'] if core_bundle['summary']['top_sector_pressure'] else 'n/d'}.",
            f"Maior pressao setorial no target800 clean plus: {clean_bundle['summary']['top_sector_pressure'][0]['sector'] if clean_bundle['summary']['top_sector_pressure'] else 'n/d'}.",
            f"HMM challenger ultimos 63 dias: acordo com regime estrutural de {100.0 * _safe_float(clean_bundle['summary']['hmm_structural_agreement_63d']):.1f}% no universo target800 clean plus.",
        ],
        "artifacts": {
            "universe_compare_csv": str(outdir / "universe_compare.csv"),
            "core_sector_detail_csv": str(outdir / "core_sector_detail.csv"),
            "target800_sector_detail_csv": str(outdir / "target800_sector_detail.csv"),
            "core_asset_detail_csv": str(outdir / "core_asset_detail.csv"),
            "target800_asset_detail_csv": str(outdir / "target800_asset_detail.csv"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
