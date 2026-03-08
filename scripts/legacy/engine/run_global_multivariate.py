"""Pipeline global multivariado: espaÃ§o de fase, compressÃ£o, cluster e novidade."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier
from scripts.finance.yf_fetch_or_load import fetch_yfinance, unify_to_daily, save_cache


def load_local_series() -> dict[str, pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    base = ROOT / "results"
    for path in (base / "_tmp" / "yfinance_daily").glob("*.csv"):
        ticker = path.stem
        df = pd.read_csv(path)
        if "date" in df.columns and "r" in df.columns:
            series[ticker] = df[["date", "r"]].copy()
    for path in (base / "phase1_curriculum" / "real").glob("*/*_daily.csv"):
        ticker = path.stem.replace("_daily", "")
        df = pd.read_csv(path)
        if "date" in df.columns and "r" in df.columns:
            series[ticker] = df[["date", "r"]].copy()
    return series


def fetch_and_prepare(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = fetch_yfinance(ticker, start=start, end=end)
    if df is None or df.empty:
        return None
    df = unify_to_daily(df)
    save_cache(df, ROOT, ticker)
    return df[["date", "r"]].copy()


def build_multivariate_matrix(
    series: dict[str, pd.DataFrame], start: str, end: str, business_days_only: bool
) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    tickers = []
    for ticker, df in series.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "r"])
        df = df.sort_values("date")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            continue
        df = df.set_index("date")[["r"]].rename(columns={"r": ticker})
        frames.append(df)
        tickers.append(ticker)
    if not frames:
        raise RuntimeError("Nenhuma sÃ©rie vÃ¡lida encontrada.")
    merged = pd.concat(frames, axis=1, join="inner").dropna()
    if business_days_only:
        merged = merged[merged.index.dayofweek < 5]
    return merged, tickers


def takens_multivariate(data: np.ndarray, m: int, tau: int) -> np.ndarray:
    n, k = data.shape
    min_len = (m - 1) * tau + 1
    if n < min_len:
        raise ValueError("dados insuficientes para embedding multivariado.")
    out_len = n - (m - 1) * tau
    blocks = []
    for lag in range(m):
        start = (m - 1 - lag) * tau
        block = data[start : start + out_len]
        blocks.append(block)
    return np.hstack(blocks)


def compute_novelty(scores: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    novelty = np.zeros(scores.shape[0], dtype=float)
    for i, label in enumerate(labels):
        if label < 0:
            novelty[i] = 1.0
        else:
            novelty[i] = scores[i]
    if novelty.size:
        med = np.median(novelty) + 1e-12
        novelty = novelty / med
    return novelty


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline global multivariado.")
    parser.add_argument("--outdir", type=str, default="results/global_multivariate")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--tickers", nargs="+", default=[])
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--pca", type=int, default=10)
    parser.add_argument("--method", type=str, default="auto", choices=("auto", "hdbscan", "kmeans"))
    parser.add_argument("--business-days-only", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_local_series()
    if args.tickers:
        if not args.allow_downloads:
            raise RuntimeError("Downloads desativados. Use --allow-downloads.")
        iterator = tqdm(args.tickers, desc="Download", unit="ticker") if tqdm else args.tickers
        for ticker in iterator:
            df = fetch_and_prepare(ticker, args.start, args.end)
            if df is None:
                continue
            series[ticker] = df

    data_df, tickers = build_multivariate_matrix(
        series,
        args.start,
        args.end,
        business_days_only=bool(int(args.business_days_only)),
    )
    data = data_df.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    embedded = takens_multivariate(data_scaled, m=args.m, tau=args.tau)

    n_components = min(args.pca, embedded.shape[1], embedded.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embedded)

    rc = RegimeClassifier(clustering_method=args.method)
    labels = rc.cluster_states(coords, {})

    unique = np.unique(labels)
    centroids = []
    scores = np.zeros(labels.shape[0], dtype=float)
    for label in unique:
        if label < 0:
            centroids.append(np.zeros(coords.shape[1]))
            continue
        mask = labels == label
        center = coords[mask].mean(axis=0)
        centroids.append(center)
        scores[mask] = np.linalg.norm(coords[mask] - center, axis=1)
    centroids = np.vstack(centroids)
    novelty = compute_novelty(scores, labels, centroids)

    summary_rows = []
    for label in unique:
        mask = labels == label
        summary_rows.append(
            {
                "cluster": int(label),
                "count": int(mask.sum()),
                "percent": float(mask.mean() * 100.0),
                "novelty_mean": float(np.mean(novelty[mask])) if mask.any() else 0.0,
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_global.csv", index=False)
    pd.DataFrame(centroids).to_csv(out_dir / "centroids.csv", index=False)
    pd.DataFrame(
        {
            "date": data_df.index[(args.m - 1) * args.tau :],
            "cluster": labels,
            "novelty": novelty,
        }
    ).to_csv(out_dir / "novelty_timeseries.csv", index=False)

    config = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "m": args.m,
        "tau": args.tau,
        "pca_components": n_components,
        "method": args.method,
        "business_days_only": bool(int(args.business_days_only)),
        "n_samples": int(coords.shape[0]),
        "n_assets": int(len(tickers)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = out_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Relatorio Global Multivariado",
                "",
                f"- Ativos: {len(tickers)}",
                f"- Periodo: {args.start} -> {args.end}",
                f"- Embedding: m={args.m}, tau={args.tau}",
                f"- PCA: {n_components} componentes",
                f"- Metodo: {args.method}",
                f"- Dias uteis apenas: {bool(int(args.business_days_only))}",
                "",
                "Arquivos:",
                "- summary_global.csv",
                "- centroids.csv",
                "- novelty_timeseries.csv",
                "- config.json",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
