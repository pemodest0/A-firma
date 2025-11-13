from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

__all__ = ["read_csv_bundle"]


def read_csv_bundle(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "ticker" not in frame.columns:
            frame = frame.assign(ticker=path.stem.upper())
        if frame.empty:
            continue
        rows.extend(frame.to_dict(orient="records"))
    return rows
