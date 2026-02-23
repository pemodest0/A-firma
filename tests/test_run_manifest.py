from __future__ import annotations

import json
from pathlib import Path

from engine.structural.run_manifest import write_run_manifest


def test_run_manifest_contains_required_fields(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    path = write_run_manifest(
        outdir,
        script="tests/test_run_manifest.py",
        params={"seed": 23},
        paths={"output": "results/x.json"},
        gates={"gate_0_ingest": True, "gate_1_causality": False},
    )
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "run_manifest.v1"
    assert isinstance(payload.get("timestamp_utc"), str) and payload["timestamp_utc"]
    assert isinstance(payload.get("git_hash"), str) and payload["git_hash"]
    assert payload.get("params", {}).get("seed") == 23
    assert "output" in payload.get("paths", {})
    gates = payload.get("gates", {})
    assert gates.get("gate_0_ingest") == "pass"
    assert gates.get("gate_1_causality") == "fail"
