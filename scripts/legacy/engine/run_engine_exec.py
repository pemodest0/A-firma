#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    plt = None
    PdfPages = None

from engine.sanity import ensure_sorted_dates, safe_test_indices, split_hash, validate_time_split
from scripts.finance.yf_fetch_or_load import find_local_data, load_price_series, fetch_yfinance, unify_to_daily, save_cache
from engine.models.takens_knn import TakensKNN
from engine.diagnostics.predictability import compute_acf, hurst_exponent_rs, lyapunov_proxy
from engine.temporal.temporal_engine import (
    TemporalConfig,
    YearResult,
    build_temporal_report,
    compare_models,
)


ASSETS_DEFAULT = ("SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "XLK", "EEM", "BTC-USD", "^VIX")


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    weekly = df["price"].resample("W-FRI").last().dropna().to_frame()
    weekly = weekly.reset_index()
    return unify_to_daily(weekly)


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std()


def load_series(asset: str, allow_downloads: bool) -> pd.DataFrame | None:
    base_dir = ROOT
    local_candidates = find_local_data(asset, base_dir)
    if (not local_candidates) and asset.startswith("^"):
        local_candidates = find_local_data(asset.replace("^", ""), base_dir)
    df = None
    for candidate in local_candidates or []:
        df = load_price_series(candidate)
        if df is not None:
            break
    if df is None:
        if not allow_downloads:
            return None
        df = fetch_yfinance(asset)
    if df is None:
        return None
    df = unify_to_daily(df)
    save_cache(df, base_dir, asset.replace("^", ""))
    return df


def make_target(df: pd.DataFrame, target: str, freq: str) -> pd.DataFrame:
    if freq == "weekly":
        df = to_weekly(df)
    else:
        df = df.copy()
    if target == "return":
        out = df[["date", "r"]].rename(columns={"r": "target"})
    else:
        window = 20 if freq == "daily" else 12
        out = df[["date", "r"]].copy()
        out["target"] = rolling_volatility(out["r"], window)
        out = out.dropna(subset=["target"])
        out = out[["date", "target"]]
    out = out.dropna()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def predict_persist(values: np.ndarray, horizon: int) -> np.ndarray:
    pred = np.full_like(values, np.nan, dtype=float)
    if len(values) <= horizon:
        return pred
    pred[horizon:] = values[:-horizon]
    return pred


def predict_ma(values: np.ndarray, horizon: int, window: int = 5) -> np.ndarray:
    pred = np.full_like(values, np.nan, dtype=float)
    if len(values) <= horizon + window:
        return pred
    for i in range(window, len(values) - horizon + 1):
        pred[i + horizon - 1] = float(np.mean(values[i - window : i]))
    return pred


def predict_knn(
    values: np.ndarray,
    horizon: int,
    tau: int = 2,
    m: int = 4,
    k: int = 10,
    train_end: int | None = None,
) -> np.ndarray:
    pred = np.full_like(values, np.nan, dtype=float)
    if len(values) <= (m - 1) * tau + horizon + 1:
        return pred
    model = TakensKNN(tau=tau, m=m, k=k)
    train_end_idx = len(values) - horizon - 1 if train_end is None else int(train_end)
    train_end_idx = max((m - 1) * tau + 1, min(train_end_idx, len(values) - horizon - 1))
    if not model.fit(values, train_end_idx):
        return pred
    for i in range((m - 1) * tau, len(values) - horizon):
        state = np.array([values[i - j * tau] for j in range(m)], dtype=float)
        pred_val = model.predict_1step(state)
        if pred_val is None:
            continue
        pred[i + horizon] = pred_val
    return pred


def predict_markov(values: np.ndarray, horizon: int, n_bins: int = 7, train_end: int | None = None) -> np.ndarray:
    pred = np.full_like(values, np.nan, dtype=float)
    if len(values) < horizon + 5:
        return pred
    train_end_idx = len(values) - horizon - 1 if train_end is None else int(train_end)
    train_end_idx = max(2, min(train_end_idx, len(values) - horizon - 1))
    train_values = np.asarray(values[: train_end_idx + 1], dtype=float)
    if train_values.size < 5:
        return pred
    quantiles = np.quantile(train_values, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(np.asarray(values, dtype=float), quantiles[1:-1], right=True)
    centers = []
    for i in range(n_bins):
        mask = bins[: train_end_idx + 1] == i
        centers.append(float(np.mean(train_values[mask])) if mask.any() else float(np.mean(train_values)))
    centers = np.array(centers)

    trans = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(train_end_idx):
        trans[bins[i], bins[i + 1]] += 1
    row_sums = trans.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    trans = trans / row_sums[:, None]

    for i in range(len(bins) - horizon):
        state = bins[i]
        mat = np.linalg.matrix_power(trans, horizon)
        probs = mat[state]
        pred[i + horizon] = float(np.dot(probs, centers))
    return pred


def evaluate_model(values: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    err = np.abs(pred[mask] - values[mask])
    return float(np.mean(err)) if err.size else float("nan")


def plot_error_by_horizon(rows: pd.DataFrame, out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, group in rows.groupby("model"):
        grouped = group.groupby("horizon")["mae"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=model)
    ax.set_xlabel("Horizonte")
    ax.set_ylabel("MAE mÃ©dio")
    ax.set_title("Erro por horizonte (mÃ©dia dos anos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_vs_pred(dates, values, pred, out_path: Path, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, values, color="black", label="real")
    ax.plot(dates, pred, color="red", label="previsto")
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executor do Motor de Analise de Serie Temporal (batch).")
    parser.add_argument("--assets", default=",".join(ASSETS_DEFAULT))
    parser.add_argument("--freqs", default="daily,weekly")
    parser.add_argument("--targets", default="return,volatility")
    parser.add_argument("--models", default="persist,ma5,knn_phase,markov_phase")
    parser.add_argument("--horizons", default="1,5,20,60")
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument("--train-window", type=int, default=5)
    parser.add_argument("--outdir", default="results/engine_exec")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()

    assets = tuple(a.strip() for a in args.assets.split(",") if a.strip())
    freqs = tuple(f.strip() for f in args.freqs.split(",") if f.strip())
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start_time = time.time()
    total_jobs = (
        len(assets)
        * len(freqs)
        * len(targets)
        * len(models)
        * len(horizons)
        * len(years)
    )
    completed = 0
    last_update = start_time

    try:
        from tqdm import tqdm

        progress = tqdm(total=total_jobs, desc="Simulacoes", unit="job")
    except Exception:
        progress = None

    for asset in assets:
        df = load_series(asset, args.allow_downloads)
        if df is None:
            continue
        for freq in freqs:
            for target in targets:
                series_df = make_target(df, target, freq)
                ensure_sorted_dates(series_df["date"])
                dates = pd.to_datetime(series_df["date"])
                values = series_df["target"].to_numpy()
                acf_summary = compute_acf(values)
                hurst_summary = hurst_exponent_rs(values)

                for year in years:
                    train_start = pd.Timestamp(f"{year - args.train_window}-01-01")
                    train_end = pd.Timestamp(f"{year - 1}-12-31")
                    test_start = pd.Timestamp(f"{year}-01-01")
                    test_end = pd.Timestamp(f"{year}-12-31")

                    mask_train = (dates >= train_start) & (dates <= train_end)
                    mask_test = (dates >= test_start) & (dates <= test_end)
                    validate_time_split(dates, mask_train, mask_test, train_end=train_end, test_start=test_start, test_end=test_end)

                    test_start_pos = int(np.where(dates >= test_start)[0].min()) if np.any(dates >= test_start) else None
                    if test_start_pos is None:
                        continue
                    train_idx = np.where(mask_train)[0]
                    if len(train_idx) == 0:
                        continue
                    train_end_pos = int(train_idx.max())

                    for horizon in horizons:
                        min_valid = test_start_pos + horizon
                        test_idx, dropped = safe_test_indices(mask_test, min_valid)
                        if len(test_idx) == 0:
                            continue
                        test_mask = np.zeros(len(values), dtype=bool)
                        test_mask[test_idx] = True

                        baseline_pred = predict_persist(values, horizon)
                        baseline_mae = evaluate_model(values, baseline_pred, test_mask)

                        for model in models:
                            if model == "persist":
                                pred = baseline_pred
                            elif model == "ma5":
                                pred = predict_ma(values, horizon, window=5)
                            elif model == "knn_phase":
                                pred = predict_knn(values, horizon, train_end=train_end_pos)
                            elif model == "markov_phase":
                                pred = predict_markov(values, horizon, train_end=train_end_pos)
                            else:
                                continue

                            mae = evaluate_model(values, pred, test_mask)
                            rows.append(
                                {
                                    "asset": asset,
                                    "freq": freq,
                                    "target": target,
                                    "model": model,
                                    "horizon": horizon,
                                    "year": year,
                                    "mae": mae,
                                    "baseline_mae": baseline_mae,
                                    "acf1": acf_summary.acf1,
                                    "hurst": hurst_summary.hurst,
                                    "split_hash": split_hash(np.where(mask_train)[0], test_idx),
                                    "dropped_test_points": dropped,
                                }
                            )
                            completed += 1
                            if progress is not None:
                                progress.update(1)
                            else:
                                now = time.time()
                                if now - last_update >= 5:
                                    rate = completed / (now - start_time)
                                    remaining = total_jobs - completed
                                    eta = remaining / rate if rate > 0 else float("inf")
                                    print(
                                        f"Progresso: {completed}/{total_jobs} "
                                        f"({completed/total_jobs:.1%}) | ETA ~ {eta/60:.1f} min"
                                    )
                                    last_update = now

                asset_rows = pd.DataFrame(rows)
                asset_rows = asset_rows[asset_rows["asset"] == asset]
                if not asset_rows.empty:
                    plot_error_by_horizon(
                        asset_rows,
                        plot_dir / f"{asset}_{freq}_{target}_error_by_horizon.png",
                    )

    overview = pd.DataFrame(rows)
    overview_path = outdir / "overview.csv"
    overview.to_csv(overview_path, index=False)

    overview_md = outdir / "overview.md"
    overview_md.write_text(
        "# Motor de Analise de Serie Temporal - Overview\n\n"
        f"Total de execuÃ§Ãµes: {len(overview)}\n",
        encoding="utf-8",
    )

    pdf_path = outdir / "overview.pdf"
    if plt is not None and PdfPages is not None:
        with PdfPages(pdf_path) as pdf:
            if overview.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis("off")
                ax.text(0.1, 0.5, "Sem resultados para gerar relatÃ³rio.", fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
            else:
                for asset in overview["asset"].unique():
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.axis("off")
                    ax.text(0.1, 0.9, f"{asset}", fontsize=14, va="top")
                    pdf.savefig(fig)
                    plt.close(fig)
                    for plot in plot_dir.glob(f"{asset}_*.png"):
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.axis("off")
                        ax.imshow(plt.imread(plot))
                        ax.set_title(plot.name.replace("_", " ").replace(".png", ""))
                        pdf.savefig(fig)
                        plt.close(fig)

    status_json = outdir / "temporal_status.json"
    if not overview.empty:
        cfg = TemporalConfig()
        model_map = {}
        for model in overview["model"].unique():
            results = []
            for year in overview["year"].unique():
                subset = overview[(overview["model"] == model) & (overview["year"] == year)]
                if subset.empty:
                    continue
                model_error = float(subset["mae"].mean())
                baseline_error = float(subset["baseline_mae"].mean())
                results.append(YearResult(year=int(year), model_error=model_error, baseline_error=baseline_error))
            model_map[model] = {"default": results}
        summaries = compare_models(model_map, cfg)
        status_report = build_temporal_report(summaries, cfg)
        status_json.write_text(json.dumps(status_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if progress is not None:
        progress.close()

    elapsed = time.time() - start_time
    print(f"Finalizado em {elapsed:.2f}s. Resultados em {outdir}")


if __name__ == "__main__":
    main()
