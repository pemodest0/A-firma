from __future__ import annotations

DEFERRED_REVIEW_TICKERS = {
    "CCZ",
    "ZD",
}


def is_deferred_review_ticker(ticker: str) -> bool:
    return str(ticker or "").strip().upper() in DEFERRED_REVIEW_TICKERS
