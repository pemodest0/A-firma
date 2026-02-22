from __future__ import annotations

from pathlib import Path

from tools.anti_leakage_guard import scan_paths, scan_text


def test_scan_text_flags_negative_shift() -> None:
    code = "x = df['ret'].shift(-1)\n"
    violations = scan_text(text=code, path="tmp/test.py")
    assert len(violations) == 1
    assert violations[0].rule == "negative_shift"


def test_scan_text_respects_allow_marker() -> None:
    code = "x = df['ret'].shift(-1)  # anti-leakage: allow\n"
    violations = scan_text(text=code, path="tmp/test.py")
    assert violations == []


def test_scan_paths_detects_centered_rolling(tmp_path: Path) -> None:
    code_dir = tmp_path / "src"
    code_dir.mkdir(parents=True, exist_ok=True)
    bad = code_dir / "bad.py"
    bad.write_text("y = s.rolling(20, center=True).mean()\n", encoding="utf-8")
    good = code_dir / "good.py"
    good.write_text("y = s.rolling(20, center=False).mean()\n", encoding="utf-8")

    violations = scan_paths(paths=[code_dir], extensions=["py"])
    assert len(violations) == 1
    assert violations[0].rule == "rolling_center_true"
    assert violations[0].path.endswith("bad.py")
