from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_MODEL_DIR = _ROOT / "modelos"
_CORE_DIR = _MODEL_DIR / "core"

for path in (_MODEL_DIR, _CORE_DIR):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
