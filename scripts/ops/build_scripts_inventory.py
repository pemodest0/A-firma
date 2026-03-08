#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
TEXT_DIRS = [
    ROOT / "config",
    ROOT / "contracts",
    ROOT / "docs",
    ROOT / "engine",
    ROOT / "execution",
    ROOT / "features",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "tools",
    ROOT / "website-ui",
]
TEXT_SUFFIXES = {
    ".cmd",
    ".json",
    ".md",
    ".plist",
    ".ps1",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


CLASS_BY_DIR = {
    "ops": "official",
    "structural": "official",
    "data": "official",
    "realestate": "official",
    "lab": "mixed",
    "finance": "mixed",
    "bench": "research",
    "legacy": "legacy",
    "sim": "research",
    "engine": "research",
    "research": "research",
    "maintenance": "maintenance",
    "report": "mixed",
    "utils": "mixed",
}


def head_short() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def classify(path: Path) -> str:
    rel = path.relative_to(SCRIPTS_ROOT).as_posix()
    if rel == "__init__.py":
        return "mixed"
    top = rel.split("/", 1)[0]
    return CLASS_BY_DIR.get(top, "review")


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for base in TEXT_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or ".next" in path.parts or "node_modules" in path.parts:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files.append(path)
    return files


def build_reference_counts(script_paths: list[Path]) -> dict[str, dict[str, int]]:
    refs = {}
    needles_by_rel: dict[str, tuple[str, ...]] = {}
    for path in script_paths:
        rel = path.relative_to(ROOT).as_posix()
        refs[rel] = {"ref_total": 0, "ref_tests": 0, "ref_launchers": 0}
        module_ref = rel.removesuffix(".py").replace("/", ".")
        needles = [rel]
        if module_ref != rel:
            needles.append(module_ref)
        needles_by_rel[rel] = tuple(needles)
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT).as_posix()
        is_test = rel.startswith("tests/")
        is_launcher = path.suffix.lower() in {".sh", ".ps1", ".cmd", ".plist"}
        for target_rel, needles in needles_by_rel.items():
            if rel == target_rel:
                continue
            if not any(needle in text for needle in needles):
                continue
            refs[target_rel]["ref_total"] += 1
            if is_test:
                refs[target_rel]["ref_tests"] += 1
            if is_launcher:
                refs[target_rel]["ref_launchers"] += 1
    return refs


def suggested_action(classification: str, ref_total: int, ref_tests: int, ref_launchers: int, rel: str) -> str:
    if classification == "official":
        return "keep_core"
    if classification == "maintenance":
        return "keep_maintenance" if ref_total or ref_tests or ref_launchers else "review_manual"
    if classification == "legacy":
        return "keep_legacy"
    if classification == "mixed":
        return "keep_active" if ref_total or ref_tests or ref_launchers else "review_manual"
    if classification == "research":
        if ref_tests or ref_launchers or ref_total:
            return "keep_research"
        if rel.startswith("scripts/bench/portfolio/") or rel.startswith("scripts/research/") or rel.startswith("scripts/sim/"):
            return "archive_candidate"
        return "review_manual"
    return "review_manual"


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / "ops" / "scripts_inventory" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    script_paths = sorted(SCRIPTS_ROOT.rglob("*.py"))
    ref_counts = build_reference_counts(script_paths)
    for p in script_paths:
        rel = p.relative_to(ROOT).as_posix()
        cls = classify(p)
        counts = ref_counts[rel]
        action = suggested_action(
            classification=cls,
            ref_total=counts["ref_total"],
            ref_tests=counts["ref_tests"],
            ref_launchers=counts["ref_launchers"],
            rel=rel,
        )
        rows.append(
            {
                "path": rel,
                "class": cls,
                "ref_total": counts["ref_total"],
                "ref_tests": counts["ref_tests"],
                "ref_launchers": counts["ref_launchers"],
                "suggested_action": action,
                "size_bytes": p.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    csv_path = out_dir / "scripts_inventory.csv"
    md_path = out_dir / "scripts_inventory.md"
    summary_path = out_dir / "summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "class",
                "ref_total",
                "ref_tests",
                "ref_launchers",
                "suggested_action",
                "size_bytes",
                "mtime_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_class = Counter(row["class"] for row in rows)
    by_action = Counter(row["suggested_action"] for row in rows)

    md_lines = [
        "# Scripts Inventory",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- head: `{head_short()}`",
        "",
        "## Count by class",
        "",
        "| class | count |",
        "|---|---:|",
    ]
    for cls in sorted(by_class):
        md_lines.append(f"| `{cls}` | {by_class[cls]} |")
    md_lines.extend(["", "## Count by suggested action", "", "| action | count |", "|---|---:|"])
    for action in sorted(by_action):
        md_lines.append(f"| `{action}` | {by_action[action]} |")
    md_lines.extend(["", "## Archive candidates", ""])
    archive_rows = [r for r in rows if r["suggested_action"] == "archive_candidate"]
    if archive_rows:
        for row in archive_rows[:120]:
            md_lines.append(
                f"- `{row['path']}` refs={row['ref_total']} tests={row['ref_tests']} launchers={row['ref_launchers']}"
            )
    else:
        md_lines.append("- (none)")
    md_lines.extend(["", "## Manual review", ""])
    review_rows = [r for r in rows if r["suggested_action"] == "review_manual"]
    if review_rows:
        for row in review_rows[:120]:
            md_lines.append(
                f"- `{row['path']}` refs={row['ref_total']} tests={row['ref_tests']} launchers={row['ref_launchers']}"
            )
    else:
        md_lines.append("- (none)")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": head_short(),
        "counts": dict(by_class),
        "actions": dict(by_action),
        "total_scripts": len(rows),
        "csv": str(csv_path.relative_to(ROOT)),
        "markdown": str(md_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = ROOT / "results" / "ops" / "scripts_inventory" / "latest_inventory.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(str(summary_path.relative_to(ROOT)))
    print(f"total_scripts={len(rows)}")


if __name__ == "__main__":
    main()
