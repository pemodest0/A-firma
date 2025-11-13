#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PYTHONPATH_EXPORT = "PYTHONPATH=modelos/core:.:modelos"

def _safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _run_forecast(command_args: Sequence[str]) -> None:
    cmd = ["bash", "-lc", " ".join(command_args)]
    result = subprocess.run(cmd, check=True)


def _collect_prediction(forecast_dir: Path, ticker: str) -> dict:
    label = ticker.replace("^", "")
    metrics_path = forecast_dir / _safe_label(label) / "daily_forecast_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics não encontrado em {metrics_path}")
    df = pd.read_csv(metrics_path, parse_dates=["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    classical = df[df["mode"] == "classical"].copy()
    classical.sort_values("date", inplace=True)
    latest_row = classical.iloc[-1]

    summary = {
        "ticker": ticker,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_as_of": latest_row["date"].date().isoformat(),
        "forecast_for": (latest_row["date"] + pd.Timedelta(days=1)).date().isoformat(),
        "classical": {
            "price_pred": float(latest_row["price_pred"]),
            "expected_return": float(latest_row["expected_return"]),
            "alpha": float(latest_row["alpha"]),
            "entropy": float(latest_row["entropy"]),
        },
        "quantum_hadamard": {
            "price_pred": float(latest_row.get("quantum_hadamard_price_pred", float("nan"))),
            "expected_return": float(latest_row.get("quantum_hadamard_expected_return", float("nan"))),
            "alpha": float(latest_row.get("quantum_hadamard_alpha", float("nan"))),
            "entropy": float(latest_row.get("quantum_hadamard_entropy", float("nan"))),
        },
        "quantum_grover": {
            "price_pred": float(latest_row.get("quantum_grover_price_pred", float("nan"))),
            "expected_return": float(latest_row.get("quantum_grover_expected_return", float("nan"))),
            "alpha": float(latest_row.get("quantum_grover_alpha", float("nan"))),
            "entropy": float(latest_row.get("quantum_grover_entropy", float("nan"))),
        },
        "vol_ratio": float(latest_row.get("vol_ratio", float("nan"))),
        "drawdown_long": float(latest_row.get("drawdown_long", float("nan"))),
        "momentum_10": float(latest_row.get("momentum_10", float("nan"))),
        "phase": latest_row.get("phase"),
        "coin_risk_flag": bool(latest_row.get("coin_risk_flag", False)),
    }
    return summary


def _load_config(config_path: Optional[str]) -> List[Dict[str, object]]:
    if not config_path:
        return []
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrada: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Config precisa conter lista 'assets'.")
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa previsões diárias e gera resumo para o próximo dia útil.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Lista de tickers (ex: SPY ^BVSP BTC-USD).")
    parser.add_argument("--forecast-output", type=str, default="results/live_forecast", help="Diretório onde run_daily_forecast salvará os arquivos.")
    parser.add_argument("--summary-output", type=str, default="results/live_predictions", help="Onde salvar o resumo diário.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--walk-steps", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--start", type=str, default=None, help="Data inicial. Quando ausente usa lookback.")
    parser.add_argument("--days-back", type=int, default=750, help="Número de dias para retroceder quando --start não é informado.")
    parser.add_argument("--end", type=str, default=None, help="Data final (padrão: hoje).")
    parser.add_argument("--forecast-extra-args", nargs=argparse.REMAINDER, default=[], help="Argumentos adicionais repassados ao run_daily_forecast (coloque após '--' na chamada).")
    parser.add_argument("--use-local-csv", action="store_true", help="Usa CSV em dados/brutos/yf/<ticker>.csv ao invés de baixar do yfinance.")
    parser.add_argument("--config", type=str, default=None, help="Arquivo JSON com configuração por ativo.")
    args = parser.parse_args()

    today = date.today()
    start = args.start or (today - timedelta(days=args.days_back)).isoformat()
    end = args.end or today.isoformat()
    forecast_output = Path(args.forecast_output)
    summary_output = Path(args.summary_output)
    summary_output.mkdir(parents=True, exist_ok=True)

    asset_configs = _load_config(args.config)
    if asset_configs:
        tasks = []
        for cfg in asset_configs:
            ticker = cfg.get("ticker")
            if not ticker:
                raise ValueError("Cada item do config precisa de 'ticker'.")
            tasks.append(
                {
                    "ticker": ticker,
                    "start": cfg.get("start", start),
                    "end": cfg.get("end", end),
                    "days_back": cfg.get("days_back", args.days_back),
                    "use_local_csv": cfg.get("use_local_csv", args.use_local_csv),
                    "csv_path": cfg.get("csv"),
                    "forecast_args": cfg.get("forecast_args", list(args.forecast_extra_args)),
                }
            )
    else:
        tasks = []
        for ticker in args.tickers:
            tasks.append(
                {
                    "ticker": ticker,
                    "start": start,
                    "end": end,
                    "days_back": args.days_back,
                    "use_local_csv": args.use_local_csv,
                    "csv_path": None,
                    "forecast_args": list(args.forecast_extra_args),
                }
            )

    results: List[dict] = []
    for task in tasks:
        ticker = task["ticker"]
        forecast_output.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHONPATH_EXPORT,
            "python3",
            "scripts/run_daily_forecast.py",
        ]
        csv_path = None
        if task.get("csv_path"):
            csv_path = Path(task["csv_path"])
        elif task.get("use_local_csv", False):
            csv_path = Path("dados/brutos/yf") / f"{ticker.replace('^', '')}.csv"
        if csv_path is not None:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV local não encontrado: {csv_path}")
            cmd.extend(["--csv", str(csv_path)])
        else:
            cmd.extend(["--symbol", ticker])
        local_start = task.get("start") or (today - timedelta(days=task.get("days_back", args.days_back))).isoformat()
        local_end = task.get("end") or end
        cmd.extend(
            [
                "--start",
                local_start,
                "--end",
                local_end,
                "--window",
                str(args.window),
                "--bins",
                str(args.bins),
                "--walk-steps",
                str(args.walk_steps),
                "--forecast-days",
                "1",
                "--noise",
                str(args.noise),
                "--output",
                str(forecast_output),
            ]
        )
        extra_args = task.get("forecast_args", [])
        if extra_args:
            cmd.extend(extra_args)
        _run_forecast(cmd)
        summary = _collect_prediction(forecast_output, ticker)
        results.append(summary)

    run_label = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = summary_output / f"live_predictions_{run_label}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Resumo salvo em {out_path}")


if __name__ == "__main__":
    main()
