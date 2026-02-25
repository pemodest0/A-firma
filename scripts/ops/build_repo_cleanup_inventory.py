#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

KEEP_TOP = {
    ".github",
    "config",
    "contracts",
    "data",
    "docs",
    "engine",
    "features",
    "models",
    "scripts",
    "tests",
    "tools",
    "website-ui",
}

REVIEW_HINTS = {
    "scripts/sim",
    "scripts/engine",
    "scripts/bench",
    "docs/notes",
}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def run_bytes(cmd: list[str]) -> bytes:
    return subprocess.check_output(cmd, cwd=ROOT)


def git_tracked_files() -> list[str]:
    out = run_bytes(["git", "ls-files", "-z"])
    if not out:
        return []
    return [p.decode("utf-8", errors="replace") for p in out.split(b"\x00") if p]


def top_dir(path: str) -> str:
    return path.split("/", 1)[0]


def classification(path: str) -> str:
    top = top_dir(path)
    if top not in KEEP_TOP:
        return "review"
    for hint in REVIEW_HINTS:
        if path == hint or path.startswith(f"{hint}/"):
            return "review"
    return "keep"


def build_report() -> dict:
    tracked = git_tracked_files()
    counts = Counter(top_dir(p) for p in tracked)

    per_top = []
    for top, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        cls = "keep" if top in KEEP_TOP else "review"
        per_top.append({"top": top, "tracked_files": n, "classification": cls})

    review_paths = [p for p in tracked if classification(p) == "review"]
    review_sample = review_paths[:120]

    untracked_top = []
    for child in sorted(ROOT.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir() and child.name not in counts:
            untracked_top.append(child.name)

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": run(["git", "rev-parse", "--short", "HEAD"]),
        "top_level_summary": per_top,
        "review_paths_sample": review_sample,
        "untracked_top_dirs": untracked_top,
        "policy": {
            "keep_top": sorted(KEEP_TOP),
            "review_hints": sorted(REVIEW_HINTS),
            "note": "Itens em review exigem dono e justificativa de permanencia.",
        },
    }


def render_md(report: dict) -> str:
    lines = []
    lines.append("# Repo Cleanup Inventory")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- head: `{report['head']}`")
    lines.append("")
    lines.append("## Top-level (tracked)")
    lines.append("")
    lines.append("| top | tracked_files | class |")
    lines.append("|---|---:|---|")
    for row in report["top_level_summary"]:
        lines.append(f"| `{row['top']}` | {row['tracked_files']} | `{row['classification']}` |")
    lines.append("")
    lines.append("## Review sample")
    lines.append("")
    if report["review_paths_sample"]:
        for p in report["review_paths_sample"]:
            lines.append(f"- `{p}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Untracked top-level dirs")
    lines.append("")
    if report["untracked_top_dirs"]:
        for d in report["untracked_top_dirs"]:
            lines.append(f"- `{d}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append("- keep_top: " + ", ".join(f"`{x}`" for x in report["policy"]["keep_top"]))
    lines.append("- review_hints: " + ", ".join(f"`{x}`" for x in report["policy"]["review_hints"]))
    lines.append("- note: " + report["policy"]["note"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / "ops" / "repo_cleanup" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "inventory.json"
    md_path = out_dir / "inventory.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")

    latest = ROOT / "results" / "ops" / "repo_cleanup" / "latest_inventory.json"
    latest.write_text(
        json.dumps(
            {
                "generated_at_utc": report["generated_at_utc"],
                "head": report["head"],
                "path": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(json_path.relative_to(ROOT)))
    print(str(md_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
