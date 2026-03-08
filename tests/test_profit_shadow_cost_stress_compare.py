from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "validation" / "run_profit_shadow_cost_stress_compare.py"
_SPEC = importlib.util.spec_from_file_location("run_profit_shadow_cost_stress_compare", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_parse_candidate_accepts_relative_spec() -> None:
    label, path = _MODULE._parse_candidate("v2=results/ops/profit_shadow_target_800_attack/runs/x/profiles/p")

    assert label == "v2"
    assert path.is_absolute()
