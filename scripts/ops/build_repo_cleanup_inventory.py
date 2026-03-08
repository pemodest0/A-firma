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
    "execution",
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
    "output",
}

TOP_LEVEL_ACTIONS = {
    ".github": "keep_core",
    "config": "keep_core",
    "contracts": "keep_core",
    "data": "keep_data",
    "docs": "keep_docs",
    "engine": "keep_core",
    "execution": "keep_core",
    "features": "keep_contract",
    "models": "keep_optional",
    "output": "archive_candidate",
    "results": "generated_only",
    "scripts": "mixed_review",
    "tests": "keep_core",
    "tools": "keep_core",
    "website-ui": "keep_core",
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


def dir_size_kb(path: Path) -> int:
    proc = subprocess.run(
        ["du", "-sk", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return int(proc.stdout.split()[0])


def git_status_rows() -> list[dict[str, str]]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:]
        rows.append({"status": status, "path": path, "top": top_dir(path)})
    return rows


def build_report() -> dict:
    tracked = git_tracked_files()
    counts = Counter(top_dir(p) for p in tracked)
    status_rows = git_status_rows()
    status_by_top = Counter(row["top"] for row in status_rows)
    status_by_kind = Counter(row["status"].strip() or "??" for row in status_rows)

    per_top = []
    for child in sorted(ROOT.iterdir(), key=lambda p: p.name):
        if child.name.startswith("."):
            continue
        top = child.name
        tracked_n = counts.get(top, 0)
        status_n = status_by_top.get(top, 0)
        on_disk = child.exists()
        size_kb = dir_size_kb(child) if on_disk else 0
        cls = "keep" if top in KEEP_TOP else "review"
        action = TOP_LEVEL_ACTIONS.get(top, "review_manual")
        per_top.append({"top": top, "classification": cls})
        per_top[-1].update(
            {
                "tracked_files": tracked_n,
                "dirty_paths": status_n,
                "size_mb": round(size_kb / 1024, 1),
                "suggested_action": action,
            }
        )

    review_paths = [p for p in tracked if classification(p) == "review"][:120]

    scripts_focus = {
        "scripts/ops": "keep_core",
        "scripts/lab": "keep_core",
        "scripts/bench/validation": "keep_research_active",
        "scripts/structural": "keep_core",
        "scripts/legacy": "keep_legacy",
        "scripts/data": "keep_support",
        "scripts/energy": "keep_domain",
        "scripts/agro": "keep_domain",
        "scripts/realestate": "keep_domain",
        "scripts/sim": "archive_candidate",
        "scripts/engine": "archive_candidate",
        "scripts/research": "archive_candidate",
        "scripts/bench/portfolio": "archive_candidate",
    }

    focus_rows = []
    for rel, action in scripts_focus.items():
        path = ROOT / rel
        if not path.exists():
            continue
        size_mb = round(dir_size_kb(path) / 1024, 1)
        file_count = sum(1 for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        focus_rows.append(
            {
                "path": rel,
                "file_count": file_count,
                "size_mb": size_mb,
                "suggested_action": action,
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": run(["git", "rev-parse", "--short", "HEAD"]),
        "top_level_summary": per_top,
        "scripts_focus": focus_rows,
        "review_paths_sample": review_paths,
        "git_status_summary": {
            "dirty_total": len(status_rows),
            "by_status": dict(status_by_kind),
            "by_top": dict(status_by_top),
        },
        "policy": {
            "keep_top": sorted(KEEP_TOP),
            "review_hints": sorted(REVIEW_HINTS),
            "note": "Itens em review exigem dono e justificativa de permanencia. Itens generated_only devem permanecer fora do commit.",
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
    lines.append("| top | tracked_files | dirty_paths | size_mb | class | action |")
    lines.append("|---|---:|---:|---:|---|---|")
    for row in report["top_level_summary"]:
        lines.append(
            f"| `{row['top']}` | {row['tracked_files']} | {row['dirty_paths']} | {row['size_mb']} | `{row['classification']}` | `{row['suggested_action']}` |"
        )
    lines.append("")
    lines.append("## Scripts focus")
    lines.append("")
    lines.append("| path | file_count | size_mb | action |")
    lines.append("|---|---:|---:|---|")
    for row in report["scripts_focus"]:
        lines.append(
            f"| `{row['path']}` | {row['file_count']} | {row['size_mb']} | `{row['suggested_action']}` |"
        )
    lines.append("")
    lines.append("## Review sample")
    lines.append("")
    if report["review_paths_sample"]:
        for p in report["review_paths_sample"]:
            lines.append(f"- `{p}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Git status summary")
    lines.append("")
    lines.append(f"- dirty_total: `{report['git_status_summary']['dirty_total']}`")
    lines.append("- by_status: " + ", ".join(f"`{k}`={v}" for k, v in sorted(report["git_status_summary"]["by_status"].items())))
    lines.append("- by_top: " + ", ".join(f"`{k}`={v}" for k, v in sorted(report["git_status_summary"]["by_top"].items())))
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
