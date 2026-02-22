#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "engine",
    "scripts/ops",
    "scripts/bench/validation",
)
DEFAULT_EXTENSIONS = (".py", ".ts", ".tsx", ".js", ".jsx")
DEFAULT_ALLOW_MARKER = "anti-leakage: allow"

_SKIP_DIRS = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass
class Rule:
    name: str
    pattern: re.Pattern[str]
    message: str


@dataclass
class Violation:
    path: str
    line: int
    rule: str
    snippet: str
    message: str


RULES = (
    Rule(
        name="negative_shift",
        pattern=re.compile(r"\.shift\(\s*-\s*\d+\s*\)", flags=re.MULTILINE),
        message="Negative shift introduces lookahead in causal pipelines.",
    ),
    Rule(
        name="negative_diff",
        pattern=re.compile(r"\.diff\(\s*-\s*\d+\s*\)", flags=re.MULTILINE),
        message="Negative diff uses future samples.",
    ),
    Rule(
        name="rolling_center_true",
        pattern=re.compile(r"\.rolling\([\s\S]{0,220}?center\s*=\s*True", flags=re.MULTILINE),
        message="Centered rolling windows are non-causal.",
    ),
)


def _resolve_path(path_raw: str) -> Path:
    path = Path(path_raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _iter_source_files(paths: Iterable[Path], extensions: set[str]) -> list[Path]:
    out: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix in extensions:
                out.append(base)
            continue
        for file in base.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix not in extensions:
                continue
            if any(part in _SKIP_DIRS for part in file.parts):
                continue
            out.append(file)
    return sorted(set(out))


def _line_number(text: str, idx: int) -> int:
    return int(text.count("\n", 0, max(0, idx)) + 1)


def _line_at(text: str, line: int) -> str:
    lines = text.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return ""


def scan_text(
    *,
    text: str,
    path: str,
    allow_marker: str = DEFAULT_ALLOW_MARKER,
) -> list[Violation]:
    out: list[Violation] = []
    marker = str(allow_marker).strip()
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            line = _line_number(text, match.start())
            snippet = _line_at(text, line)
            if marker and marker in snippet:
                continue
            out.append(
                Violation(
                    path=path,
                    line=line,
                    rule=rule.name,
                    snippet=snippet,
                    message=rule.message,
                )
            )
    return out


def scan_paths(
    *,
    paths: Iterable[Path],
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    allow_marker: str = DEFAULT_ALLOW_MARKER,
) -> list[Violation]:
    ext_set = {e if str(e).startswith(".") else f".{e}" for e in extensions}
    files = _iter_source_files(paths, extensions=ext_set)
    violations: list[Violation] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file.read_text(encoding="latin-1")
        except OSError:
            continue
        rel = str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)
        violations.extend(scan_text(text=text, path=rel, allow_marker=allow_marker))
    return violations


def main() -> None:
    ap = argparse.ArgumentParser(description="Static anti-lookahead guard for production code paths.")
    ap.add_argument(
        "--paths",
        type=str,
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Relative or absolute paths to scan.",
    )
    ap.add_argument(
        "--extensions",
        type=str,
        default=",".join(e.lstrip(".") for e in DEFAULT_EXTENSIONS),
        help="Comma-separated extensions, e.g. py,ts,tsx.",
    )
    ap.add_argument("--allow-marker", type=str, default=DEFAULT_ALLOW_MARKER)
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON payload.")
    args = ap.parse_args()

    path_values = [str(p).strip() for p in (args.paths or []) if str(p).strip()]
    if not path_values:
        path_values = list(DEFAULT_PATHS)
    paths = [_resolve_path(p) for p in path_values]
    ext_values = [x.strip() for x in str(args.extensions).split(",") if x.strip()]
    if not ext_values:
        ext_values = [e.lstrip(".") for e in DEFAULT_EXTENSIONS]

    violations = scan_paths(paths=paths, extensions=ext_values, allow_marker=str(args.allow_marker))

    payload = {
        "status": "ok" if not violations else "fail",
        "n_violations": int(len(violations)),
        "paths": [str(p) for p in path_values],
        "extensions": [str(e) for e in ext_values],
        "allow_marker": str(args.allow_marker),
        "violations": [asdict(v) for v in violations],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"[anti_leakage_guard] status={payload['status']} "
            f"n_violations={payload['n_violations']} paths={','.join(path_values)}"
        )
        for v in violations:
            print(f"- {v.path}:{v.line} [{v.rule}] {v.message} :: {v.snippet}")
    if violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
