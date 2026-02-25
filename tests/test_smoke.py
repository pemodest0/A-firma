from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def test_pipeline_smoke(tmp_path: Path) -> None:
    start = datetime(2025, 1, 1)
    times = [start + timedelta(hours=i) for i in range(48)]
    values = [100 + i * 0.1 for i in range(48)]
    df = pd.DataFrame({"timestamp": times, "load": values})

    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    outdir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "engine.run",
            "--input",
            str(input_csv),
            "--time-col",
            "timestamp",
            "--value-col",
            "load",
            "--outdir",
            str(outdir),
            "--pdf",
        ],
        check=True,
    )

    assert (outdir / "processed.csv").exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "report.pdf").exists()
