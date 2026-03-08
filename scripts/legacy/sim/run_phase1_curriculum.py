"""Fase 1: gera sÃ©ries sintÃ©ticas + dados reais para treinamento/benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier
from scripts.finance.yf_fetch_or_load import fetch_yfinance, unify_to_daily, save_cache


def logistic_map(r: float, n: int, x0: float = 0.2, discard: int = 1000) -> np.ndarray:
    x = np.zeros(n + discard)
    x[0] = x0
    for i in range(1, n + discard):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[discard:]


def henon_map(a: float, b: float, n: int, x0: float = 0.1, y0: float = 0.1, discard: int = 1000) -> np.ndarray:
    x = np.zeros(n + discard)
    y = np.zeros(n + discard)
    x[0], y[0] = x0, y0
    for i in range(1, n + discard):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x[discard:]


def write_summary_for_label(series: np.ndarray, label: str, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = RegimeClassifier(tau=1, m=2, clustering_method="hdbscan")
    embedded = rc.embed(series)
    velocity = rc.compute_velocity(series)
    energy = rc.compute_energy(embedded[:, 0], velocity)
    labels = np.full(embedded.shape[0], label, dtype=object)
    summary = rc._build_summary(embedded, velocity, energy, labels)
    pd.DataFrame(summary).to_csv(out_dir / f"summary_{name}.csv", index=False)
    pd.DataFrame({"x": series}).to_csv(out_dir / f"{name}.csv", index=False)


def run_real_series(
    ticker: str,
    out_dir: Path,
    start: str | None,
    end: str | None,
    allow_downloads: bool,
    no_plots: bool,
) -> None:
    if not allow_downloads:
        raise RuntimeError("Downloads desativados. Use --allow-downloads.")
    df = fetch_yfinance(ticker, start=start or "2009-01-01", end=end)
    if df is None or df.empty:
        raise RuntimeError(f"Falha ao baixar {ticker}.")
    df = unify_to_daily(df)
    save_cache(df, ROOT, ticker)
    series = df["r"].to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{ticker}_daily.csv", index=False)

    rc = RegimeClassifier(clustering_method="hdbscan")
    rc.run_full_analysis(
        series=series,
        output_dir=out_dir,
        system_type="auto",
        filename_suffix=f"_{ticker}",
        smooth_labels=True,
        min_run=3,
        generate_plots=not no_plots,
        generate_report=not no_plots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 do curriculum (sinteticas + reais).")
    parser.add_argument("--outdir", type=str, default="results/phase1_curriculum")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Nao gerar plots/relatorios.")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "XLK", "EEM", "BTC-USD", "^VIX"],
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    # Logistic map (periodico vs caotico)
    for r in [3.2, 3.5, 3.6, 3.7, 3.8, 3.9]:
        label = "periodico" if r < 3.57 else "caotico"
        tasks.append(("logistic", {"r": r, "label": label}))

    # Henon map (parametros com regimes distintos)
    tasks.append(("henon", {"a": 1.2, "b": 0.3, "label": "periodico"}))
    tasks.append(("henon", {"a": 1.3, "b": 0.3, "label": "periodico"}))
    tasks.append(("henon", {"a": 1.4, "b": 0.3, "label": "caotico"}))
    tasks.append(("henon", {"a": 1.4, "b": 0.2, "label": "caotico"}))

    # Reais
    for ticker in args.tickers:
        tasks.append(("real", {"ticker": ticker}))

    iterator = tqdm(tasks, desc="Phase1", unit="task") if tqdm else tasks
    for task_type, info in iterator:
        if tqdm:
            iterator.set_postfix(task=task_type)
        start_time = time.time()
        if task_type == "logistic":
            series = logistic_map(info["r"], n=5000)
            name = f"logistic_r{info['r']}".replace(".", "_")
            write_summary_for_label(series, info["label"], out_dir / "synthetic", name)
        elif task_type == "henon":
            series = henon_map(info["a"], info["b"], n=5000)
            name = f"henon_a{info['a']}_b{info['b']}".replace(".", "_")
            write_summary_for_label(series, info["label"], out_dir / "synthetic", name)
        elif task_type == "real":
            ticker = info["ticker"]
            run_real_series(
                ticker,
                out_dir / "real" / ticker.replace("^", ""),
                args.start,
                args.end,
                args.allow_downloads,
                args.no_plots,
            )
        elapsed = time.time() - start_time
        if not tqdm:
            print(f"[done] {task_type} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

