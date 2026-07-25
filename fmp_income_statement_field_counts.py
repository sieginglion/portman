#!/usr/bin/env python3
"""Count fields returned by FMP for the most liquid US tickers."""

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from typing import Any

import httpx
from dotenv import load_dotenv

import quadrant

FMP_INCOME_STATEMENT_URL = (
    "https://financialmodelingprep.com/stable/income-statement-as-reported"
)
MIN_FIELD_OCCURRENCES = quadrant.RESULT_LIMIT // 8


def count_data_fields(response_data: Any) -> Counter[str]:
    """Return field occurrence counts from each response row's ``data`` object."""
    if not isinstance(response_data, list):
        raise ValueError(
            f"expected a list response, got {type(response_data).__name__}"
        )

    counts: Counter[str] = Counter()
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
        counts.update(data.keys())
    return counts


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
            counts.update(count_data_fields(response_data))
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            message = str(exc)
            print(f"Skipping {ticker}: {message}", file=sys.stderr)

    return counts


def load_liquid_us_tickers(api_key: str, pool_size: int | None) -> list[str]:
    pool_size = pool_size or quadrant.candidate_pool_size(
        quadrant.RESULT_LIMIT, quadrant.CANDIDATE_POOL_MULTIPLIER
    )
    candidates = quadrant.load_top_candidates(
        market=quadrant.MARKET_CONFIG_BY_CODE["u"],
        api_key=api_key,
        top_n_symbols=pool_size,
        top_n_results=quadrant.RESULT_LIMIT,
    )
    return [candidate.symbol for candidate in candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count fields in one quarterly FMP as-reported income statement "
        "for the most liquid US tickers."
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        help="number of US tickers to rank for liquidity (default: quadrant pool size)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("FMP_API_KEY") or os.environ.get("FMP_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY (or FMP_KEY) is required")

    tickers = load_liquid_us_tickers(api_key, args.pool_size)

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
