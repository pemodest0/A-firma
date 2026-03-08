from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "run_profit_method_failure_audit.py"
_SPEC = importlib.util.spec_from_file_location("run_profit_method_failure_audit", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_safe_float_handles_bad_values() -> None:
    assert _MODULE._safe_float("1.25") == 1.25
    assert str(_MODULE._safe_float("bad")).lower() == "nan"
