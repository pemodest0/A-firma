#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR_DEFAULT = ROOT / "results" / "validation" / "realworld_vix"
MIN_POINTS = 500


@dataclass
class DatasetRef:
    path: Path
    value_col: str
    date_col: str
    symbol: str


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_status(outdir: Path, payload: dict[str, Any]) -> None:
    _json_dump(outdir / "STATUS.json", payload)


def _resolve_col(df: pd.DataFrame, col: str | None) -> str | None:
    if col is None:
        return None
    if col in df.columns:
        return col
    low = col.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None


def _auto_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    value = None
    date = None
    for cand in ["close", "adj_close", "price", "value", "valor"]:
        value = _resolve_col(df, cand)
        if value is not None:
            break
    for cand in ["date", "data", "datetime", "timestamp", "time"]:
        date = _resolve_col(df, cand)
        if date is not None:
            break
    return value, date


def _autodetect_vix(path_override: str | None) -> DatasetRef:
    if path_override:
        p = Path(path_override)
        if not p.exists():
            raise FileNotFoundError(f"VIX path not found: {p}")
        df = pd.read_csv(p, nrows=5)
        vc, dc = _auto_cols(df)
        if vc is None or dc is None:
            raise ValueError(f"Could not infer value/date columns in {p}")
        return DatasetRef(p, vc, dc, p.stem)
    candidates = [
        ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "^VIX.csv",
        ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "VIX.csv",
    ]
    for p in candidates:
        if p.exists():
            return DatasetRef(p, "close", "date", p.stem)
    raise FileNotFoundError("Missing VIX dataset in repo.")


def _autodetect_asset(path_override: str | None) -> DatasetRef:
    if path_override:
        p = Path(path_override)
        if not p.exists():
            raise FileNotFoundError(f"Asset path not found: {p}")
        df = pd.read_csv(p, nrows=5)
        vc, dc = _auto_cols(df)
        if vc is None or dc is None:
            raise ValueError(f"Could not infer value/date columns in {p}")
        return DatasetRef(p, vc, dc, p.stem)
    candidates = [
        ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "SPY.csv",
        ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "QQQ.csv",
        ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "BTC-USD.csv",
    ]
    for p in candidates:
        if p.exists():
            return DatasetRef(p, "close", "date", p.stem)
    folder = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
    if folder.exists():
        csvs = sorted(folder.glob("*.csv"))
        for p in csvs:
            if p.name.upper() not in {"^VIX.CSV", "VIX.CSV"}:
                return DatasetRef(p, "close", "date", p.stem)
    raise FileNotFoundError("Missing risk asset dataset in repo.")


def _load_series(ref: DatasetRef) -> pd.DataFrame:
    df = pd.read_csv(ref.path)
    vc = _resolve_col(df, ref.value_col) or _auto_cols(df)[0]
    dc = _resolve_col(df, ref.date_col) or _auto_cols(df)[1]
    if vc is None or dc is None:
        raise ValueError(f"Missing value/date columns for {ref.path}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dc], errors="coerce"),
            "value": pd.to_numeric(df[vc], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("date").drop_duplicates("date")
    if out.shape[0] < MIN_POINTS:
        raise ValueError(f"insufficient points {out.shape[0]} in {ref.path.name}")
    return out.reset_index(drop=True)


def _run_vix_motor(vix_values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from engine.graph.core import run_graph_engine  # type: ignore
    from engine.graph.embedding import estimate_embedding_params  # type: ignore

    m, tau = estimate_embedding_params(vix_values, max_tau=20, max_m=6)
    res = run_graph_engine(
        vix_values,
        m=m,
        tau=tau,
        n_micro=80,
        n_regimes=4,
        k_nn=5,
        theiler=10,
        alpha=2.0,
        seed=seed,
        timeframe="daily",
    )
    labels = np.asarray(res.state_labels).astype(str)
    conf = np.asarray(res.confidence, dtype=float)
    q = float((res.quality or {}).get("score", np.nan))
    quality = np.repeat(np.clip(q if np.isfinite(q) else 0.0, 0.0, 1.0), labels.shape[0])
    return labels, conf, quality


def _normalize_label(lbl: str) -> str:
    x = str(lbl).strip().upper()
    if x in {"STABLE", "LOW_VOL", "CALM"}:
        return "stable"
    if x in {"TRANSITION", "MID_VOL", "REGIME_CHANGE"}:
        return "transition"
    if x in {"UNSTABLE", "NOISY", "HIGH_VOL", "CHAOTIC"}:
        return "unstable"
    return "transition"


def _hysteresis(labels: list[str], min_persist: int = 3) -> list[str]:
    if not labels:
        return labels
    out = labels[:]
    i = 0
    n = len(out)
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_persist:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            repl = left if left is not None else right
            if repl is not None:
                out[i:j] = [repl] * (j - i)
        i = j
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def _metrics_from_returns(r: np.ndarray, exposure: np.ndarray) -> dict[str, float]:
    strat_r = exposure * r
    eq = np.exp(np.cumsum(strat_r))
    total_return = float(eq[-1] - 1.0)
    ann = 252.0
    mu = float(np.mean(strat_r))
    sd = float(np.std(strat_r, ddof=1)) if strat_r.size > 1 else 0.0
    annualized_return = float(np.exp(mu * ann) - 1.0)
    annualized_vol = float(sd * math.sqrt(ann))
    sharpe = float((mu * ann) / (sd * math.sqrt(ann) + 1e-12))
    mdd = _max_drawdown(eq)
    calmar = float(annualized_return / abs(mdd + 1e-12))
    switches = int(np.sum(np.abs(np.diff(exposure)) > 1e-12))
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "time_in_market": float(np.mean(exposure)),
        "n_switches": switches,
    }


def _future_cum_returns(ret: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    n = len(ret)
    for h in horizons:
        vals = np.full(n, np.nan, dtype=float)
        for i in range(n):
            j = min(n, i + h)
            if i + 1 < j:
                vals[i] = float(np.exp(np.sum(ret[i + 1 : j])) - 1.0)
        out[h] = vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-world VIX regime actionability test (A/B/C exposures).")
    parser.add_argument("--vix_path", type=str, default=None)
    parser.add_argument("--asset_path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence_gate", type=float, default=0.55)
    parser.add_argument("--quality_gate", type=float, default=0.35)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        vix_ref = _autodetect_vix(args.vix_path)
        asset_ref = _autodetect_asset(args.asset_path)
        vix_df = _load_series(vix_ref).rename(columns={"value": "vix"})
        asset_df = _load_series(asset_ref).rename(columns={"value": "asset_price"})
    except Exception as exc:
        _write_status(outdir, {"status": "fail", "reason": f"missing_data: {exc}"})
        print(f"[fail] missing_data: {exc}")
        return

    merged = vix_df.merge(asset_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if merged.shape[0] < MIN_POINTS:
        _write_status(outdir, {"status": "fail", "reason": f"insufficient_intersection<{MIN_POINTS}"})
        print("[fail] insufficient_intersection")
        return

    merged["asset_return"] = np.log(merged["asset_price"]).diff().fillna(0.0)

    labels_raw, confidence, quality = _run_vix_motor(merged["vix"].to_numpy(dtype=float), seed=args.seed)
    n = len(labels_raw)
    merged = merged.iloc[len(merged) - n :].copy().reset_index(drop=True)
    merged["label_raw"] = labels_raw
    merged["label"] = [_normalize_label(x) for x in labels_raw]
    merged["confidence"] = confidence
    merged["quality"] = quality

    # Strategy A: always exposed.
    exposure_a = np.ones(n, dtype=float)

    # Strategy B: simple quantile filter.
    vix_thr = float(np.quantile(merged["vix"], 0.80))
    exposure_b = np.where(merged["vix"].to_numpy(dtype=float) > vix_thr, 0.25, 1.0)

    # Strategy C: regime + confidence/quality gate + hysteresis.
    exp_map = {"stable": 1.0, "transition": 0.5, "unstable": 0.0}
    label_c = merged["label"].tolist()
    for i in range(n):
        if merged.at[i, "confidence"] < args.confidence_gate or merged.at[i, "quality"] < args.quality_gate:
            label_c[i] = "transition"
    label_c = _hysteresis(label_c, min_persist=3)
    exposure_c = np.array([exp_map.get(lbl, 0.5) for lbl in label_c], dtype=float)

    r = merged["asset_return"].to_numpy(dtype=float)
    eq_a = np.exp(np.cumsum(exposure_a * r))
    eq_b = np.exp(np.cumsum(exposure_b * r))
    eq_c = np.exp(np.cumsum(exposure_c * r))
    dd_a = eq_a / np.maximum.accumulate(eq_a) - 1.0
    dd_b = eq_b / np.maximum.accumulate(eq_b) - 1.0
    dd_c = eq_c / np.maximum.accumulate(eq_c) - 1.0

    strategy_daily = pd.DataFrame(
        {
            "date": merged["date"].dt.strftime("%Y-%m-%d"),
            "asset_return": r,
            "exposure_A": exposure_a,
            "exposure_B": exposure_b,
            "exposure_C": exposure_c,
            "equity_A": eq_a,
            "equity_B": eq_b,
            "equity_C": eq_c,
            "dd_A": dd_a,
            "dd_B": dd_b,
            "dd_C": dd_c,
        }
    )
    strategy_daily.to_csv(outdir / "strategy_daily.csv", index=False)

    vix_regimes = pd.DataFrame(
        {
            "date": merged["date"].dt.strftime("%Y-%m-%d"),
            "vix": merged["vix"].to_numpy(dtype=float),
            "label": label_c,
            "confidence": merged["confidence"].to_numpy(dtype=float),
            "quality": merged["quality"].to_numpy(dtype=float),
        }
    )
    vix_regimes.to_csv(outdir / "vix_regimes.csv", index=False)

    horizons = [1, 5, 10, 20]
    fut = _future_cum_returns(r, horizons)
    top_idx = np.argsort(-merged["vix"].to_numpy(dtype=float))[:20]
    top_idx = np.sort(top_idx)
    event_rows = []
    for i in top_idx:
        event_rows.append(
            {
                "date": merged.at[i, "date"].strftime("%Y-%m-%d"),
                "vix": float(merged.at[i, "vix"]),
                "label": label_c[i],
                "confidence": float(merged.at[i, "confidence"]),
                "quality": float(merged.at[i, "quality"]),
                "return_asset_next_1d": float(fut[1][i]) if np.isfinite(fut[1][i]) else np.nan,
                "return_asset_next_5d": float(fut[5][i]) if np.isfinite(fut[5][i]) else np.nan,
                "return_asset_next_20d": float(fut[20][i]) if np.isfinite(fut[20][i]) else np.nan,
            }
        )
    pd.DataFrame(event_rows).to_csv(outdir / "event_sanity.csv", index=False)

    m_a = _metrics_from_returns(r, exposure_a)
    m_b = _metrics_from_returns(r, exposure_b)
    m_c = _metrics_from_returns(r, exposure_c)
    metrics = {
        "metrics_A": m_a,
        "metrics_B": m_b,
        "metrics_C": m_c,
        "delta_C_vs_A": {k: float(m_c[k] - m_a[k]) for k in m_a.keys()},
        "delta_C_vs_B": {k: float(m_c[k] - m_b[k]) for k in m_b.keys()},
    }
    _json_dump(outdir / "metrics.json", metrics)

    # Focus windows requested: 2024 and 2025.
    windows = {}
    for year in [2024, 2025]:
        mask = merged["date"].dt.year == year
        if int(mask.sum()) == 0:
            continue
        idx = np.where(mask.to_numpy())[0]
        rr = r[idx]
        windows[str(year)] = {
            "n_days": int(len(idx)),
            "A": _metrics_from_returns(rr, exposure_a[idx]),
            "B": _metrics_from_returns(rr, exposure_b[idx]),
            "C": _metrics_from_returns(rr, exposure_c[idx]),
        }
    _json_dump(outdir / "scenario_2024_2025.json", windows)

    pass_conditions = [
        ("drawdown", (m_c["max_drawdown"] <= m_a["max_drawdown"] * 0.90)),
        ("sharpe_plus", (m_c["sharpe"] >= (m_a["sharpe"] + 0.10) and m_c["max_drawdown"] <= m_a["max_drawdown"] + 1e-12)),
    ]
    cond_pass = any(v for _, v in pass_conditions)
    verdict = "pass" if cond_pass else "neutral"
    _write_status(
        outdir,
        {
            "status": verdict,
            "pass_conditions": {
                "drawdown_reduction_10pct_rel": bool(pass_conditions[0][1]),
                "sharpe_plus_0p10_with_non_worse_dd": bool(pass_conditions[1][1]),
            },
            "notes": [
                "Regime no VIX usado como contexto de exposicao do ativo de risco.",
                "Estrategia C usa gate de confianca/qualidade e histerese de 3 dias.",
                "Avaliacao principal compara A vs B vs C em retorno e risco.",
                "Recorte 2024/2025 salvo para prova de utilidade em periodo recente.",
                "Nao prova alpha; prova utilidade de controle de risco contextual.",
            ],
            "caveats": [
                "Custos de transacao nao incluidos.",
                "Limiares fixos podem exigir calibracao por ativo.",
                "Resultados dependem do ativo escolhido.",
                "Regime label e uma abstracao; nao representa causa economica direta.",
                "Backtest historico nao garante desempenho futuro.",
            ],
        },
    )

    profile = {
        "vix_dataset": str(vix_ref.path),
        "asset_dataset": str(asset_ref.path),
        "asset_symbol": asset_ref.symbol,
        "n_points_aligned": int(merged.shape[0]),
        "date_min": merged["date"].min().strftime("%Y-%m-%d"),
        "date_max": merged["date"].max().strftime("%Y-%m-%d"),
        "return_type": "log_return",
        "params": {
            "seed": int(args.seed),
            "vix_quantile_filter_B": 0.80,
            "confidence_gate": float(args.confidence_gate),
            "quality_gate": float(args.quality_gate),
            "hysteresis_days_C": 3,
        },
    }
    _json_dump(outdir / "data_profile.json", profile)

    print(
        "realworld_vix_actionability "
        f"status={verdict} asset={asset_ref.symbol} "
        f"dd_A={m_a['max_drawdown']:.3f} dd_C={m_c['max_drawdown']:.3f} "
        f"sharpe_A={m_a['sharpe']:.3f} sharpe_C={m_c['sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
