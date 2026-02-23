#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_fetch_module() -> Any:
    mod_path = ROOT / "scripts" / "data" / "fetch_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_datasets_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable_to_load_module:{mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _parse_years(text: str) -> list[int]:
    vals: list[int] = []
    for token in str(text or "").split(","):
        x = token.strip()
        if not x:
            continue
        vals.append(int(x))
    return sorted(set(vals))


def _selected_years(cfg_years: list[int], from_year: int | None, to_year: int | None, years_csv: str) -> list[int]:
    if years_csv.strip():
        years = _parse_years(years_csv)
    elif cfg_years:
        years = sorted(set(int(y) for y in cfg_years))
    else:
        now_y = datetime.now(timezone.utc).year
        years = [now_y]
    if from_year is not None:
        years = [y for y in years if y >= int(from_year)]
    if to_year is not None:
        years = [y for y in years if y <= int(to_year)]
    return years


def _dataset_source_files(raw_root: Path, source: str, dataset: str) -> list[Path]:
    d = raw_root / source / dataset
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.csv") if p.is_file())


def _looks_yearly_file(path: Path) -> bool:
    stem = path.stem
    years = []
    for token in stem.replace("-", "_").split("_"):
        if len(token) == 4 and token.isdigit():
            years.append(token)
    return len(years) == 1 and stem.endswith(years[0])


def _file_meta(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(st.st_size),
        "sha256": _sha256(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot sync for ONS energy data + canonical pack build.")
    ap.add_argument("--source", default="ONS")
    ap.add_argument("--datasets", default="ons_carga_diaria,ons_curva_carga_horaria")
    ap.add_argument("--from-year", type=int, default=None)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--years", default="")
    ap.add_argument("--download", type=int, default=1, help="1=attempt official download, 0=use local raw only")
    ap.add_argument("--force", type=int, default=0, help="1=redownload existing files")
    ap.add_argument("--combine", type=int, default=1, help="1=build combined CSV per dataset")
    ap.add_argument("--build-pack", type=int, default=1, help="1=build canonical normalized energy pack")
    ap.add_argument("--run-adequacy", type=int, default=1, help="1=run data adequacy gate after pack")
    ap.add_argument("--pack-start", default="2018-01-01")
    ap.add_argument("--pack-end", default="")
    ap.add_argument("--pack-min-rows", type=int, default=300)
    ap.add_argument("--business-days-only", type=int, default=0)
    ap.add_argument("--results-dir", default="results/energy_download")
    args = ap.parse_args()

    run_id = f"energy_sync_{_utc_id()}"
    outdir = ROOT / str(args.results_dir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    fetch_mod = _load_fetch_module()
    config = fetch_mod.load_config()
    source = str(args.source).strip()
    source_cfg = config.get(source) or {}
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    raw_root = ROOT / "data" / "raw"

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "download_enabled": bool(int(args.download)),
        "force_download": bool(int(args.force)),
        "datasets": [],
        "pack": {},
        "adequacy": {},
        "status": "ok",
    }

    for dataset in datasets:
        dataset_cfg = source_cfg.get(dataset) or {}
        cfg_years = [int(y) for y in (dataset_cfg.get("years") or [])]
        years = _selected_years(cfg_years, args.from_year, args.to_year, str(args.years))
        row: dict[str, Any] = {
            "dataset": dataset,
            "years_target": years,
            "downloads": [],
            "errors": [],
        }
        downloaded_now: list[Path] = []

        if bool(int(args.download)):
            for year in years:
                url_template = str(dataset_cfg.get("url_template", ""))
                filename = Path(url_template.format(year=year)).name if url_template else f"{dataset}_{year}.csv"
                expected_path = raw_root / source / dataset / filename
                existed_before = expected_path.exists()
                got = fetch_mod.fetch_dataset(source, dataset, int(year), bool(int(args.force)))
                status = "missing"
                if got is not None:
                    downloaded_now.append(Path(got))
                    if existed_before and not bool(int(args.force)):
                        status = "skipped_existing"
                    elif existed_before and bool(int(args.force)):
                        status = "redownloaded"
                    else:
                        status = "downloaded"
                else:
                    row["errors"].append(f"download_failed:{dataset}:{year}")
                row["downloads"].append(
                    {
                        "year": int(year),
                        "status": status,
                        "path": str(expected_path.relative_to(ROOT)),
                    }
                )

        files_now = _dataset_source_files(raw_root, source, dataset)
        row["files_now"] = [_file_meta(p) for p in files_now]

        combine_candidates = sorted(set(downloaded_now))
        if not combine_candidates:
            combine_candidates = [p for p in files_now if _looks_yearly_file(p)]
        if bool(int(args.combine)) and len(combine_candidates) >= 2:
            combined_out = outdir / f"{dataset}_{years[0]}_{years[-1]}.csv"
            try:
                fetch_mod._combine_csvs(combine_candidates, combined_out)
                row["combined_file"] = _file_meta(combined_out)
            except Exception as exc:  # noqa: BLE001
                row["errors"].append(f"combine_failed:{exc}")

        if row["errors"]:
            manifest["status"] = "partial"
        manifest["datasets"].append(row)

    if bool(int(args.build_pack)):
        pack_cmd = [
            PY,
            "scripts/data/build_local_energy_pack.py",
            "--raw-dir",
            "data/raw/ONS/ons_carga_diaria",
            "--results-dir",
            str(args.results_dir),
            "--start",
            str(args.pack_start),
            "--end",
            str(args.pack_end),
            "--min-rows",
            str(int(args.pack_min_rows)),
            "--business-days-only",
            str(int(args.business_days_only)),
            "--write-canonical-raw",
            "1",
        ]
        code, out, err = _run(pack_cmd)
        manifest["pack"] = {
            "cmd": pack_cmd,
            "code": int(code),
            "status": "ok" if code == 0 else "fail",
            "stdout_tail": out[-4000:],
            "stderr_tail": err[-4000:],
        }
        if code != 0:
            manifest["status"] = "fail"

    if bool(int(args.run_adequacy)):
        adequacy_out = ROOT / "results" / "validation" / f"data_adequacy_{run_id}"
        adequacy_cmd = [
            PY,
            "scripts/bench/validation/16_data_adequacy_gate.py",
            "--outdir",
            str(adequacy_out),
        ]
        code, out, err = _run(adequacy_cmd)
        summary_path = adequacy_out / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                summary = {}
        manifest["adequacy"] = {
            "cmd": adequacy_cmd,
            "code": int(code),
            "status": "ok" if code == 0 else "fail",
            "summary": summary,
            "stdout_tail": out[-4000:],
            "stderr_tail": err[-4000:],
        }
        if code != 0 and manifest["status"] == "ok":
            manifest["status"] = "partial"

    out_path = outdir / "sync_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = ROOT / str(args.results_dir) / "latest_sync.json"
    latest.write_text(json.dumps({"run_id": run_id, "manifest": str(out_path)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "run_id": run_id, "manifest": str(out_path)}, ensure_ascii=False))

    if manifest["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
