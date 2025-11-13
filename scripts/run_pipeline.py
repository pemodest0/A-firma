#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
PYTHON_EXECUTABLE = Path(sys.executable).as_posix()
PYTHONPATH_EXPORT = "PYTHONPATH=modelos/core:.:modelos"


def _safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _run(cmd: List[str], cwd: Path) -> None:
    print("Executando:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao executar {' '.join(cmd)} (rc={result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestra ETL → previsão → residual → benchmark.")
    parser.add_argument("config", type=str, help="Arquivo JSON com definição de ativos e parâmetros.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config não encontrada: {config_path}")

    with config_path.open() as fh:
        config: Dict[str, object] = json.load(fh)

    project_root = Path(config.get("project_root", ".")).resolve()
    assets: List[Dict[str, object]] = config.get("assets", [])
    benchmark_output = config.get("benchmark_output", "results/multi_asset_benchmark")
    years = config.get("benchmark_years", [2024, 2025])
    log_dir = Path(config.get("log_dir", "results/pipeline_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"Pipeline iniciado às {datetime.now().isoformat()}\n")

    for asset in assets:
        label = asset["label"]
        data_path = asset["data"]
        forecast_output = asset.get("forecast_output", f"results/{label.lower()}_forecast")
        window = str(asset.get("window", 30))
        bins = str(asset.get("bins", 15))
        start = asset.get("start")
        end = asset.get("end")

        # 1. Data quality
        dq_output = asset.get("dq_output")
        dq_args = [PYTHONPATH_EXPORT, PYTHON_EXECUTABLE, "scripts/check_data_quality.py", data_path]
        if dq_output:
            dq_args.extend(["--output", dq_output])
        _run(["bash", "-lc", " ".join(dq_args)], project_root)

        # 2. Forecast
        forecast_cmd = [
            PYTHONPATH_EXPORT,
            PYTHON_EXECUTABLE,
            "scripts/run_daily_forecast.py",
            "--csv",
            data_path,
            "--window",
            window,
            "--bins",
            bins,
            "--output",
            forecast_output,
        ]
        if start:
            forecast_cmd.extend(["--start", start])
        if end:
            forecast_cmd.extend(["--end", end])
        extra_args = asset.get("forecast_extra_args")
        if isinstance(extra_args, list):
            forecast_cmd.extend(str(arg) for arg in extra_args)
        _run(["bash", "-lc", " ".join(forecast_cmd)], project_root)

        # 3. Residual
        metrics_path = str(
            Path(forecast_output)
            / _safe_label(Path(data_path).stem or label)
            / "daily_forecast_metrics.csv"
        )
        if asset.get("residual_enabled", True):
            residual_output = asset.get("residual_output", f"results/hybrid_residual_{label.lower()}")
            residual_cmd = [
                PYTHONPATH_EXPORT,
                PYTHON_EXECUTABLE,
                "scripts/train_residual_model.py",
                "--metrics",
                    metrics_path,
                    "--start",
                    str(asset.get("residual_start", start or "2010-01-01")),
                    "--train-end",
                    str(asset.get("residual_train_end", end or "2023-12-31")),
                "--test-start",
                str(asset.get("residual_test_start", "2024-01-01")),
                "--output",
                residual_output,
            ]
            target_mode = asset.get("residual_target_mode")
            if target_mode:
                residual_cmd.extend(["--target-mode", str(target_mode)])
            if asset.get("residual_tune", True):
                residual_cmd.append("--tune")
                splits = asset.get("residual_tune_splits")
                if splits is not None:
                    residual_cmd.extend(["--tune-splits", str(splits)])
            else:
                print(f"[INFO] Residual (tune) desativado para {label}.")
            _run(["bash", "-lc", " ".join(residual_cmd)], project_root)
        else:
            print(f"[INFO] Residual desativado para {label}.")

    # Benchmark multiativos
    assets_args = []
    for asset in assets:
        label = asset["label"].upper()
        forecast_output = asset.get("forecast_output", f"results/{label.lower()}_forecast")
        metrics_path = (
            Path(forecast_output)
            / _safe_label(Path(asset["data"]).stem if isinstance(asset.get("data"), str) else asset["label"])
            / "daily_forecast_metrics.csv"
        )
        assets_args.append(f"{label}:{metrics_path}")

        benchmark_cmd = [
            PYTHONPATH_EXPORT,
        PYTHON_EXECUTABLE,
        "scripts/benchmark_multi_assets.py",
        "--assets",
        " ".join(assets_args),
        "--years",
        " ".join(str(year) for year in years),
        "--output",
        str(benchmark_output),
    ]
    _run(["bash", "-lc", " ".join(benchmark_cmd)], project_root)

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"Pipeline finalizado às {datetime.now().isoformat()}\n")

    print(f"Pipeline concluído. Logs em {log_file}")


if __name__ == "__main__":
    main()
