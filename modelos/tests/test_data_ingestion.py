import pandas as pd
from pathlib import Path

from data_ingestion import normalize_price_csv


def test_normalize_price_csv_handles_european_format(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    raw.write_text("""Date;Price
10/01/2024;123,456
11/01/2024;124,789
12/01/2024;125,321
""")
    result = normalize_price_csv(raw, separator=";", decimal=",", thousands=".")
    frame = result.frame
    assert list(frame.columns) == ["date", "price"]
    assert frame.shape[0] == 3
    assert frame["price"].iloc[1] == 124.789


def test_normalize_price_csv_respects_manual_columns(tmp_path: Path):
    raw = tmp_path / "raw2.csv"
    raw.write_text("""DATA;Último
2024-01-01;10,50
2024-01-02;11,25
""")
    result = normalize_price_csv(raw, date_column="DATA", price_column="Último", separator=";", decimal=",")
    frame = result.frame
    assert frame["price"].tolist() == [10.5, 11.25]
