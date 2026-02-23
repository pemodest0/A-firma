from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List

REPO_ROOT = Path(__file__).resolve().parents[2]


def _detect_delimiter(sample_line: str) -> str:
    if ";" in sample_line and "," not in sample_line:
        return ";"
    if "," in sample_line and ";" not in sample_line:
        return ","
    return ";"


def _combine_csvs(paths: List[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("No CSVs to combine.")
    header = None
    delimiter = None
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for idx, path in enumerate(paths):
            with path.open("r", encoding="utf-8") as infile:
                first_line = infile.readline()
                if not first_line:
                    continue
                if delimiter is None:
                    delimiter = _detect_delimiter(first_line)
                if header is None:
                    header = first_line.strip()
                    handle.write(header + "\n")
                else:
                    if first_line.strip() != header:
                        raise ValueError(f"Inconsistent header in {path}")
                for line in infile:
                    if line.strip():
                        handle.write(line)


def load_config() -> dict:
    config_path = REPO_ROOT / "data_sources.json"
    if not config_path.exists():
        raise FileNotFoundError(f"data_sources.json not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def download_with_requests(url: str, output_path: Path) -> int:
    import requests

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
        return total


def download_with_urllib(url: str, output_path: Path) -> int:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
        output_path.write_bytes(data)
        return len(data)


def fetch_dataset(source: str, dataset: str, year: int, force: bool) -> Optional[Path]:
    config = load_config()
    source_cfg = config.get(source)
    if not source_cfg:
        raise ValueError(f"Source not found in config: {source}")

    dataset_cfg = source_cfg.get(dataset)
    if not dataset_cfg:
        raise ValueError(f"Dataset not found in config: {dataset}")

    url_template = dataset_cfg.get("url_template")
    if not url_template:
        raise ValueError(f"Missing url_template for dataset: {dataset}")

    url = url_template.format(year=year)
    filename = Path(url).name
    output_dir = REPO_ROOT / "data" / "raw" / source / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return output_path

    try:
        try:
            size = download_with_requests(url, output_path)
        except Exception:
            size = download_with_urllib(url, output_path)
        print(f"[ok] {output_path} ({size} bytes)")
        return output_path
    except Exception as exc:
        print(f"[error] Failed to download {url}: {exc}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public datasets")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--from-year", type=int)
    parser.add_argument("--to-year", type=int)
    parser.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_year and args.to_year:
        paths = []
        for year in range(args.from_year, args.to_year + 1):
            path = fetch_dataset(args.source, args.dataset, year, args.force)
            if path:
                paths.append(path)
        if args.combine and paths:
            output_dir = REPO_ROOT / "data" / "raw" / args.source / args.dataset
            output_path = output_dir / f"{args.dataset}_{args.from_year}_{args.to_year}.csv"
            _combine_csvs(paths, output_path)
            print(f"[ok] Combined CSV saved at {output_path}")
    elif args.year:
        fetch_dataset(args.source, args.dataset, args.year, args.force)
    else:
        raise ValueError("Provide --year or --from-year/--to-year.")


if __name__ == "__main__":
    main()
