import pandas as pd

from data_quality import analyze_price_series


def test_analyze_price_series_detects_issues():
    df = pd.DataFrame({
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",  # duplicate
            "2024-01-05",  # gap (missing 3,4)
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
        ],
        "price": [100, 101, 101, 0, -5, 150, 20_000],
    })

    report = analyze_price_series(df)

    assert report.duplicate_dates >= 1
    assert report.non_positive_prices >= 2
    assert report.missing_business_days >= 1
    assert report.outlier_returns >= 1
