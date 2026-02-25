#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ASSET_RE = re.compile(r"(?<![A-Z0-9])(\^?[A-Z]{2,5}(?:-[A-Z]{2,5})?)(?![A-Z0-9])")
FREQ_RE = re.compile(r"(daily|weekly)", re.IGNORECASE)
TAG_RULES = {
    "walkforward": re.compile(r"walkforward", re.IGNORECASE),
    "benchmark": re.compile(r"benchmark", re.IGNORECASE),
    "regime": re.compile(r"regime", re.IGNORECASE),
    "phase": re.compile(r"phase", re.IGNORECASE),
    "forecast": re.compile(r"forecast", re.IGNORECASE),
    "overview": re.compile(r"overview", re.IGNORECASE),
    "report": re.compile(r"report", re.IGNORECASE),
    "risk": re.compile(r"risk", re.IGNORECASE),
    "metrics": re.compile(r"metrics", re.IGNORECASE),
    "attractor": re.compile(r"attractor", re.IGNORECASE),
    "recurrence": re.compile(r"recurrence", re.IGNORECASE),
    "lorenz": re.compile(r"lorenz", re.IGNORECASE),
    "vanderpol": re.compile(r"vanderpol", re.IGNORECASE),
    "duffing": re.compile(r"duffing", re.IGNORECASE),
    "pendulo": re.compile(r"pendulo|pendulum", re.IGNORECASE),
    "engine": re.compile(r"engine", re.IGNORECASE),
    "synthetic": re.compile(r"synthetic", re.IGNORECASE),
    "global": re.compile(r"global", re.IGNORECASE),
    "dashboard": re.compile(r"dashboard", re.IGNORECASE),
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_run_id(rel_path: str) -> str:
    parts = rel_path.split(os.sep)
    return parts[0] if len(parts) > 1 else "__root__"


def detect_asset(text: str) -> str:
    for match in ASSET_RE.finditer(text):
        return match.group(1)
    return "__global__"


def detect_freq(text: str) -> Optional[str]:
    m = FREQ_RE.search(text)
    return m.group(1).lower() if m else None


def detect_tags(text: str) -> List[str]:
    tags = []
    for tag, rx in TAG_RULES.items():
        if rx.search(text):
            tags.append(tag)
    return tags


def artifact_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".svg"}:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    if ext in {".md", ".markdown"}:
        return "md"
    return "other"


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return out
    keys = {
        "mase",
        "dir_acc",
        "rmse",
        "mae",
        "roc_auc",
        "f1",
        "bal_acc",
        "accuracy",
    }
    for k in keys:
        if k in data:
            out[k] = data[k]
    for k in ("warnings", "alert", "alerts", "status"):
        if k in data:
            out[k] = data[k]
    if "metrics" in data and isinstance(data["metrics"], dict):
        for k in keys:
            if k in data["metrics"]:
                out[k] = data["metrics"][k]
    if "summary" in data and isinstance(data["summary"], dict):
        for k in keys:
            if k in data["summary"]:
                out[k] = data["summary"][k]
    return out


def read_csv_meta(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            meta["columns"] = header
            n_rows = 0
            for _ in reader:
                n_rows += 1
            meta["n_rows"] = n_rows
    except Exception:
        return meta
    return meta


def build_index(root: Path, out_path: Path) -> Dict[str, Any]:
    existing: Dict[str, Any] = {}
    existing_files: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for item in existing.get("files", []):
                existing_files[item["rel_path"]] = item
        except Exception:
            existing = {}
            existing_files = {}

    files: List[Dict[str, Any]] = []
    assets = set()
    tags = set()
    runs: Dict[str, Dict[str, Any]] = {}
    metrics_index: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            rel_path = str(path.relative_to(root))
            stat = path.stat()
            size_bytes = stat.st_size
            mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            prev = existing_files.get(rel_path)
            if prev and prev.get("size_bytes") == size_bytes and prev.get("mtime_iso") == mtime_iso:
                files.append(prev)
                assets.add(prev.get("asset", "__global__"))
                for t in prev.get("tags", []):
                    tags.add(t)
                run_id = prev.get("run_id", "__root__")
                runs.setdefault(run_id, {"files": 0, "assets": set(), "tags": set()})
                runs[run_id]["files"] += 1
                runs[run_id]["assets"].add(prev.get("asset", "__global__"))
                for t in prev.get("tags", []):
                    runs[run_id]["tags"].add(t)
                continue

            rel_lower = rel_path.lower()
            run_id = detect_run_id(rel_path)
            asset = detect_asset(rel_path)
            freq = detect_freq(rel_lower)
            tgs = detect_tags(rel_lower)
            a_type = artifact_type(path)

            entry: Dict[str, Any] = {
                "rel_path": rel_path,
                "filename": name,
                "size_bytes": size_bytes,
                "mtime_iso": mtime_iso,
                "run_id": run_id,
                "asset": asset,
                "freq": freq,
                "artifact_type": a_type,
                "tags": tgs,
            }

            if a_type == "json":
                data = safe_read_json(path)
                if data is not None:
                    entry["json_keys"] = list(data.keys()) if isinstance(data, dict) else []
                    metrics = extract_metrics(data)
                    if metrics:
                        entry["metrics"] = metrics
                        metrics_index.setdefault(run_id, {}).setdefault(asset, {})[freq or "unknown"] = metrics

            if a_type == "csv":
                entry["csv_meta"] = read_csv_meta(path)

            files.append(entry)
            assets.add(asset)
            for t in tgs:
                tags.add(t)
            runs.setdefault(run_id, {"files": 0, "assets": set(), "tags": set()})
            runs[run_id]["files"] += 1
            runs[run_id]["assets"].add(asset)
            for t in tgs:
                runs[run_id]["tags"].add(t)

    runs_serialized: Dict[str, Any] = {}
    for run_id, info in runs.items():
        runs_serialized[run_id] = {
            "files": info["files"],
            "assets": sorted(info["assets"]),
            "tags": sorted(info["tags"]),
        }

    index = {
        "generated_at": iso_now(),
        "root": str(root),
        "runs": runs_serialized,
        "files": files,
        "assets": sorted(assets),
        "tags": sorted(tags),
        "metrics": metrics_index,
    }
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Index results folder into results_index.json")
    parser.add_argument("--root", default="results", help="Path to results folder")
    parser.add_argument("--out", default="results/results_index.json", help="Output index path")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"results root not found: {root}")

    index = build_index(root, out_path)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote index: {out_path}")


if __name__ == "__main__":
    main()
