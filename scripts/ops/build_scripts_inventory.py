#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"


CLASS_BY_DIR = {
    "ops": "official",
    "structural": "official",
    "data": "official",
    "realestate": "official",
    "lab": "mixed",
    "finance": "mixed",
    "bench": "research",
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


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / "ops" / "scripts_inventory" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(SCRIPTS_ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        cls = classify(p)
        rows.append(
            {
                "path": rel,
                "class": cls,
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
            f, fieldnames=["path", "class", "size_bytes", "mtime_utc"]
        )
        writer.writeheader()
        writer.writerows(rows)

    by_class = {}
    for row in rows:
        by_class[row["class"]] = by_class.get(row["class"], 0) + 1

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
    md_lines.extend(["", "## Review list", ""])
    review = [r["path"] for r in rows if r["class"] in {"research", "review"}]
    if review:
        for p in review[:200]:
            md_lines.append(f"- `{p}`")
    else:
        md_lines.append("- (none)")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": head_short(),
        "counts": by_class,
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
