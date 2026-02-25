#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MigrationRule:
    domain: str
    source: Path
    target: Path
    recursive: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_source_files(src: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in src.rglob("*") if p.is_file())
    else:
        yield from (p for p in src.glob("*") if p.is_file())


def build_rules() -> list[MigrationRule]:
    return [
        MigrationRule(
            domain="finance",
            source=ROOT / "data" / "raw" / "finance",
            target=ROOT / "data" / "download" / "finance",
        ),
        MigrationRule(
            domain="energy",
            source=ROOT / "data" / "raw" / "ONS",
            target=ROOT / "data" / "download" / "energy" / "ons_daily",
        ),
        MigrationRule(
            domain="energy",
            source=ROOT / "data" / "raw" / "energy",
            target=ROOT / "data" / "download" / "energy" / "regional",
        ),
        MigrationRule(
            domain="realestate",
            source=ROOT / "data" / "raw" / "realestate",
            target=ROOT / "data" / "download" / "realestate",
        ),
        MigrationRule(
            domain="realestate",
            source=ROOT / "data" / "realestate" / "core",
            target=ROOT / "data" / "clean" / "realestate" / "core",
        ),
        MigrationRule(
            domain="realestate",
            source=ROOT / "data" / "realestate" / "normalized",
            target=ROOT / "data" / "processed" / "realestate" / "normalized",
        ),
    ]


def should_copy(src: Path, dst: Path) -> tuple[bool, str]:
    if not dst.exists():
        return True, "new"
    if src.stat().st_size != dst.stat().st_size:
        return True, "size_changed"
    # Same size: compare hash to avoid stale copies with same size but different content.
    if file_hash(src) != file_hash(dst):
        return True, "hash_changed"
    return False, "unchanged"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Organiza dados em camadas canonicas sem apagar legado."
    )
    ap.add_argument(
        "--domain",
        action="append",
        choices=["finance", "energy", "realestate"],
        help="Dominio para migrar. Pode repetir. Se omitido, migra todos.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Aplica copia. Sem esta flag, roda em dry-run.",
    )
    ap.add_argument(
        "--verify-hash",
        action="store_true",
        help="Gera hash SHA256 para source e target no manifesto.",
    )
    args = ap.parse_args()

    selected = set(args.domain or ["finance", "energy", "realestate"])
    rules = [r for r in build_rules() if r.domain in selected]

    ts = utc_now()
    out_dir = ROOT / "results" / "ops" / "data_layout" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int | bool]] = []
    totals = {"candidates": 0, "copied": 0, "unchanged": 0, "skipped_missing_source": 0}

    for rule in rules:
        if not rule.source.exists():
            totals["skipped_missing_source"] += 1
            manifest_rows.append(
                {
                    "domain": rule.domain,
                    "source": str(rule.source.relative_to(ROOT)),
                    "target": str(rule.target.relative_to(ROOT)),
                    "relative_path": "",
                    "action": "skip_missing_source",
                    "applied": False,
                    "reason": "source_missing",
                    "size_bytes": 0,
                    "source_sha256": "",
                    "target_sha256": "",
                }
            )
            continue

        for src_file in iter_source_files(rule.source, rule.recursive):
            rel = src_file.relative_to(rule.source)
            dst_file = rule.target / rel
            totals["candidates"] += 1

            do_copy, reason = should_copy(src_file, dst_file)
            applied = False
            action = "keep"
            if do_copy:
                action = "copy"
                if args.apply:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    applied = True
                    totals["copied"] += 1
            else:
                totals["unchanged"] += 1

            src_hash = file_hash(src_file) if args.verify_hash else ""
            dst_hash = file_hash(dst_file) if args.verify_hash and dst_file.exists() else ""

            manifest_rows.append(
                {
                    "domain": rule.domain,
                    "source": str(rule.source.relative_to(ROOT)),
                    "target": str(rule.target.relative_to(ROOT)),
                    "relative_path": str(rel),
                    "action": action,
                    "applied": applied,
                    "reason": reason,
                    "size_bytes": src_file.stat().st_size,
                    "source_sha256": src_hash,
                    "target_sha256": dst_hash,
                }
            )

    json_path = out_dir / "migration_manifest.json"
    csv_path = out_dir / "migration_manifest.csv"
    summary_path = out_dir / "summary.json"

    json_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "source",
                "target",
                "relative_path",
                "action",
                "applied",
                "reason",
                "size_bytes",
                "source_sha256",
                "target_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "apply_mode": bool(args.apply),
        "domains": sorted(selected),
        "totals": totals,
        "manifest_json": str(json_path.relative_to(ROOT)),
        "manifest_csv": str(csv_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = ROOT / "results" / "ops" / "data_layout" / "latest_summary.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(str(summary_path.relative_to(ROOT)))
    print(f"candidates={totals['candidates']} copied={totals['copied']} unchanged={totals['unchanged']}")


if __name__ == "__main__":
    main()
