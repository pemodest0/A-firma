from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = ROOT / "scripts" / "ops" / "agent_guides"
MASTER_GUIDE = GUIDES_DIR / "AGENTS_MAIN.md"


def _coerce_meta_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return text


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    lines = raw.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}
    meta: dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _coerce_meta_value(value)
    return meta


def load_agent_guide(agent_id: str) -> dict[str, Any]:
    guide_path = GUIDES_DIR / f"{agent_id}.md"
    meta = _parse_frontmatter(guide_path)
    return {
        "agent_id": str(agent_id),
        "guide_path": str(guide_path.relative_to(ROOT)) if guide_path.exists() else "",
        "master_guide_path": str(MASTER_GUIDE.relative_to(ROOT)) if MASTER_GUIDE.exists() else "",
        "display_name": str(meta.get("display_name") or agent_id),
        "mission": str(meta.get("mission") or ""),
        "run_order": meta.get("run_order"),
        "depends_on": meta.get("depends_on") if isinstance(meta.get("depends_on"), list) else [],
        "consumes": meta.get("consumes") if isinstance(meta.get("consumes"), list) else [],
        "produces": meta.get("produces") if isinstance(meta.get("produces"), list) else [],
        "rules": meta.get("rules") if isinstance(meta.get("rules"), list) else [],
    }


def attach_agent_guide(summary: dict[str, Any], agent_id: str) -> dict[str, Any]:
    payload = dict(summary)
    payload["agent"] = load_agent_guide(agent_id)
    return payload
