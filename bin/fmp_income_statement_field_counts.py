#!/usr/bin/env python3
"""Count fields returned by FMP for the most liquid US tickers."""

import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import quadrant  # noqa: E402

FMP_INCOME_STATEMENT_URL = (
    "https://financialmodelingprep.com/stable/income-statement-as-reported"
)
MIN_FIELD_OCCURRENCES = quadrant.RESULT_LIMIT // 8


def extract_data_fields(response_data: Any) -> set[str]:
    """Return the distinct fields present in a ticker response's ``data`` objects."""
    if not isinstance(response_data, list):
        raise ValueError(
            f"expected a list response, got {type(response_data).__name__}"
        )

    fields: set[str] = set()
    for row in response_data:
        if not isinstance(row, Mapping):
            raise ValueError(
                f"expected each row to be an object, got {type(row).__name__}"
            )
        data = row.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(
                f"expected row['data'] to be an object, got {type(data).__name__}"
            )
        fields.update(data.keys())
    return fields


def fetch_field_counts(tickers: list[str], api_key: str) -> Counter[str]:
    counts: Counter[str] = Counter()

    for ticker in tickers:
        try:
            response_data = quadrant.cached_get_json(
                FMP_INCOME_STATEMENT_URL,
                params=[
                    ("symbol", ticker),
                    ("period", "quarter"),
                    ("limit", 1),
                    ("apikey", api_key),
                ],
            )
            counts.update(extract_data_fields(response_data))
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            print(f"Skipping {ticker}: {exc}", file=sys.stderr)

    return counts


def load_liquid_us_tickers(api_key: str) -> list[str]:
    pool_size = quadrant.candidate_pool_size(
        quadrant.RESULT_LIMIT, quadrant.CANDIDATE_POOL_MULTIPLIER
    )
    candidates = quadrant.load_top_candidates(
        market=quadrant.MARKET_CONFIG_BY_CODE["u"],
        api_key=api_key,
        top_n_symbols=pool_size,
        top_n_results=quadrant.RESULT_LIMIT,
    )
    return [candidate.symbol for candidate in candidates]


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY") or os.environ.get("FMP_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY (or FMP_KEY) is required")

    tickers = load_liquid_us_tickers(api_key)

    counts = fetch_field_counts(tickers, api_key)
    counts = Counter(
        {
            field: count
            for field, count in counts.items()
            if count >= MIN_FIELD_OCCURRENCES
        }
    )

    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
