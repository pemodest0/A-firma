from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.shadow_health import write_shadow_health_outputs  # noqa: E402


def main() -> None:
    repo_root = ROOT
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outdir = repo_root / "results/ops/shadow_health" / run_id
    summary = write_shadow_health_outputs(repo_root, outdir=outdir)
    compact = {
        "status": summary.get("status"),
        "generated_at_utc": summary.get("generated_at_utc"),
        "summary": summary.get("summary"),
        "recommended_active_watchlist": [
            {
                "alias": item.get("alias"),
                "candidate_id": item.get("candidate_id"),
                "role": item.get("role"),
                "priority": item.get("priority"),
            }
            for item in summary.get("recommended_active_watchlist", [])
            if isinstance(item, dict)
        ],
    }
    print(compact)


if __name__ == "__main__":
    main()
