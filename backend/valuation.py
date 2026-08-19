import asyncio
import json
import math
import os
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from typing import Literal, TypeGuard

import numpy as np
import pandas as pd
from httpx import HTTPError, HTTPStatusError
from loguru import logger

from . import shared
from .shared import (
    EODHD_API_KEY,
    FINMIND_KEY,
    FINNHUB_API_KEY,
    FMP_KEY,
    MASSIVE_API_KEY,
    TIINGO_API_KEY,
    add_suffix,
)

EXTRA_Q = 1
SOURCE_DIFF_THRESHOLD = 0.065
MAX_DATE_MATCH_DIFFERENCE_DAYS = 6
EPS_ABS_TOLERANCE = 0.02
FINNHUB_MILLION_SCALE = 1_000_000

# A provider row can also contain metadata such as CIK and fiscal quarter.
type IncomeStatementRow = dict[str, object]
type SourceIncomeStatementRows = dict[str, IncomeStatementRow]
type IncomeStatementSources = dict[str, SourceIncomeStatementRows]
type IncomeStatementFetcher = Callable[[str, int], Awaitable[SourceIncomeStatementRows]]

type XpsValue = int | float | None
type SourceFieldValues = dict[str, XpsValue]
type SecQ4ValueKind = Literal["flow", "average"]
type SecFiscalQuarter = Literal["Q1", "Q2", "Q3", "Q4"]


@dataclass(frozen=True)
class SourcePeriod:
    """A provider row and the period-end date it reported."""

    date: str
    row: IncomeStatementRow


@dataclass
class AlignedQuarter:
    """Aligned field values plus source-period metadata for SEC lookup."""

    fields: dict[str, SourceFieldValues] = field(default_factory=dict)
    source_periods: dict[str, list[SourcePeriod]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SecFieldSpec:
    concept: str
    unit: str
    q4_value_kind: SecQ4ValueKind


@dataclass(frozen=True)
class SecQuarterMetadata:
    date: str
    period: SecFiscalQuarter


type AlignedQuarters = dict[str, AlignedQuarter]
type ResolvedQuarter = dict[str, XpsValue]
type ResolvedQuarters = dict[str, ResolvedQuarter]


@dataclass(frozen=True)
class Consensus:
    value: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class USIncomeStatementSource:
    name: str
    label: str
    enabled: bool
    fetcher: IncomeStatementFetcher


US_INCOME_STATEMENT_SOURCES = (
    USIncomeStatementSource(
        "fmp",
        "FMP",
        True,
        lambda symbol, limit: fetch_fmp_income_statements("u", symbol, limit),
    ),
    USIncomeStatementSource(
        "massive",
        "Massive",
        shared.ENABLE_MASSIVE_FUNDAMENTALS,
        lambda symbol, limit: fetch_massive_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        "eodhd",
        "EODHD",
        shared.ENABLE_EODHD_FUNDAMENTALS,
        lambda symbol, limit: fetch_eodhd_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        "finnhub",
        "Finnhub",
        True,
        lambda symbol, limit: fetch_finnhub_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        "tiingo",
        "Tiingo",
        shared.ENABLE_TIINGO_FUNDAMENTALS,
        lambda symbol, limit: fetch_tiingo_income_statements(symbol, limit),
    ),
)
ENABLED_US_INCOME_STATEMENT_SOURCES = tuple(
    source for source in US_INCOME_STATEMENT_SOURCES if source.enabled
)
SOURCE_LABELS = {
    **{source.name: source.label for source in US_INCOME_STATEMENT_SOURCES},
    "sec": "SEC",
}
SEC_REFERENCE_SOURCES = ("sec",)


def us_income_statement_source_order(
    include_sec: bool = False,
) -> tuple[str, ...]:
    names = tuple(source.name for source in ENABLED_US_INCOME_STATEMENT_SOURCES)
    return (*names, *SEC_REFERENCE_SOURCES) if include_sec else names


BASE_XPS_FIELDS = ("revenue", "weightedAverageShsOutDil")
EPS_XPS_FIELD = "epsDiluted"
ALL_XPS_FIELDS = (*BASE_XPS_FIELDS, EPS_XPS_FIELD)
XPS_COLUMNS = ("rps", "eps")


def xps_field_map(shares_field: str, eps_field: str) -> dict[str, str]:
    return {
        "revenue": "revenue",
        "weightedAverageShsOutDil": shares_field,
        EPS_XPS_FIELD: eps_field,
    }


def new_xps_quarter_counts() -> dict[str, dict[str, int]]:
    return {
        field: {source: 0 for source in us_income_statement_source_order()}
        for field in ALL_XPS_FIELDS
    }


def new_xps_consensus_pair_counts() -> dict[str, dict[str, int]]:
    source_names = us_income_statement_source_order()
    return {
        field: {
            f"{left}:{right}": 0
            for index, left in enumerate(source_names)
            for right in source_names[index + 1 :]
        }
        for field in ALL_XPS_FIELDS
    }


@dataclass
class XpsDiagnostics:
    seen: set[tuple[str, str]]
    coverage: dict[str, dict[str, int]]
    agreement: dict[str, dict[str, int]]


def new_xps_diagnostics() -> XpsDiagnostics:
    return XpsDiagnostics(
        seen=set(),
        coverage=new_xps_quarter_counts(),
        agreement=new_xps_consensus_pair_counts(),
    )


# Diagnostics are intentionally process-local. Restarting the backend starts a new
# screener run with clean counts. SEC is a reference source, not a peer
# provider, so it is excluded.
_xps_diagnostics = new_xps_diagnostics()
SEC_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)
SEC_COMPANY_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json"
)
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN_TERMINAL_FINANCIAL_STATEMENT_URL = (
    "https://api-internal.tokenterminal.com/trpc/projects.getFinancialStatement"
)
# The Token Terminal bearer token is copied from tt.sh. The short-lived browser
# session JWT is read from .env so it can be refreshed locally.
TOKEN_TERMINAL_AUTHORIZATION = "Bearer c0e5035a-64f6-4d2c-b5f6-ac1d1cb3da2f"
TOKEN_TERMINAL_PROJECT_SLUGS = {"UNI": "uniswap"}
TOKEN_TERMINAL_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)
EODHD_FUNDAMENTALS_URL = "https://eodhd.com/api/v1.1/fundamentals/{}"
TIINGO_FUNDAMENTALS_STATEMENTS_URL = (
    "https://api.tiingo.com/tiingo/fundamentals/{}/statements"
)
FINMIND_TAIWAN_QUARTER_BUFFER = 1
FINMIND_TAIWAN_MONTH_BUFFER = 2
TIINGO_QUARTER_BUFFER = 1
FINMIND_TAIWAN_SHARE_PAR_VALUE = 10
SEC_DEDUPE_COLS = ["filed", "val", "start", "end"]
SEC_ALLOWED_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
SEC_METADATA_SOURCES = ("fmp", "massive")
# Revenue concepts differ by filer. Prefer the contract-revenue fact and use the
# generic revenues fact only when the preferred concept is unavailable.
SEC_FIELD_SPECS = {
    "revenue": (
        SecFieldSpec(
            concept="RevenueFromContractWithCustomerExcludingAssessedTax",
            unit="USD",
            q4_value_kind="flow",
        ),
        SecFieldSpec(
            concept="Revenues",
            unit="USD",
            q4_value_kind="flow",
        ),
    ),
    "weightedAverageShsOutDil": (
        SecFieldSpec(
            concept="WeightedAverageNumberOfDilutedSharesOutstanding",
            unit="shares",
            q4_value_kind="average",
        ),
    ),
    EPS_XPS_FIELD: (
        SecFieldSpec(
            concept="EarningsPerShareDiluted",
            unit="USD/shares",
            q4_value_kind="flow",
        ),
    ),
}


def empty_sec_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(SEC_DEDUPE_COLS))


def source_log_diff(lhs: int | float, rhs: int | float) -> float:
    lhs = float(lhs)
    rhs = float(rhs)
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return math.inf
    if lhs == rhs:
        return 0.0
    if lhs == 0 or rhs == 0:
        return math.inf
    if (lhs < 0) != (rhs < 0):
        return math.inf
    return abs(math.log(abs(lhs)) - math.log(abs(rhs)))


def has_source_field_value(value: XpsValue) -> TypeGuard[int | float]:
    return value is not None and not pd.isna(value) and math.isfinite(value)


def record_xps_diagnostics(
    symbol: str,
    aligned_quarters: AlignedQuarters,
) -> None:
    """Record provider field-quarter counts and pairwise consensus."""
    diagnostics = _xps_diagnostics
    for quarter, data in aligned_quarters.items():
        key = (symbol, quarter)
        if key in diagnostics.seen:
            continue

        peer_values_by_field = {
            field_name: usable_source_values(
                data.fields.get(field_name, {}), include_sec=False
            )
            for field_name in ALL_XPS_FIELDS
        }
        if not any(peer_values_by_field.values()):
            continue

        diagnostics.seen.add(key)
        for field_name in ALL_XPS_FIELDS:
            counts = diagnostics.coverage[field_name]
            peer_values = peer_values_by_field[field_name]
            for source, _ in peer_values:
                counts[source] += 1

            pair_counts = diagnostics.agreement[field_name]
            for (left_source, left_value), (right_source, right_value) in combinations(
                peer_values, 2
            ):
                if field_has_consensus(field_name, left_value, right_value):
                    pair_counts[f"{left_source}:{right_source}"] += 1


def get_xps_diagnostics() -> dict:
    """Return provider coverage and pairwise-agreement counts."""
    return {
        "total_quarters": len(_xps_diagnostics.seen),
        "coverage": _xps_diagnostics.coverage,
        "agreement": _xps_diagnostics.agreement,
    }


def select_closest_date_key(target_date: str, date_keys: list[str]) -> str | None:
    target_ts = pd.Timestamp(target_date)
    matches = []
    for date_key in date_keys:
        delta_days = abs((pd.Timestamp(date_key) - target_ts).days)
        if delta_days > MAX_DATE_MATCH_DIFFERENCE_DAYS:
            continue
        matches.append((delta_days, date_key))

    if not matches:
        return None

    matches.sort()
    return matches[0][1]


def align_source_rows(
    aligned_quarters: AlignedQuarters,
    source_name: str,
    source_rows: SourceIncomeStatementRows,
) -> None:
    for source_date, row in sorted(source_rows.items()):
        quarter_key = select_closest_date_key(source_date, list(aligned_quarters))
        if quarter_key is None:
            quarter_key = source_date
            aligned_quarters[quarter_key] = AlignedQuarter()
        quarter = aligned_quarters[quarter_key]
        # Retain every matching row: SEC metadata uses the first one, while CIK
        # lookup may need a later row with a usable CIK.
        quarter.source_periods.setdefault(source_name, []).append(
            SourcePeriod(date=source_date, row=row)
        )
        for field_name in ALL_XPS_FIELDS:
            if field_name not in row:
                continue
            value = row[field_name]
            if not isinstance(value, (int, float)) and value is not None:
                continue
            quarter.fields.setdefault(field_name, {})[source_name] = value


def sanitize_source_field_value(
    field: str, value: int | float | None
) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if field in BASE_XPS_FIELDS and value <= 0:
        return None
    return value


def normalize_source_row(
    row: dict,
    field_map: dict[str, str],
    *,
    base_field_scale: int | float = 1,
) -> dict[str, int | float | None]:
    normalized_row = {}
    for field_name, source_field in field_map.items():
        value = row.get(source_field)
        if value is not None and field_name in BASE_XPS_FIELDS:
            value *= base_field_scale
        normalized_row[field_name] = sanitize_source_field_value(field_name, value)
    return normalized_row


def normalize_income_statement_rows(
    rows: list[dict],
    *,
    field_map: dict[str, str],
    date_field: str,
    metadata_fields: dict[str, str] | None = None,
    base_field_scale: int | float = 1,
) -> dict[str, dict]:
    normalized_rows = {}
    for row in rows:
        date = parse_timestamp(row.get(date_field))
        if date is None:
            continue

        normalized_row = normalize_source_row(
            row,
            field_map,
            base_field_scale=base_field_scale,
        )
        normalized_row.update(
            {
                output_field: row.get(source_field)
                for output_field, source_field in (metadata_fields or {}).items()
            }
        )
        normalized_rows[date_str(date)] = normalized_row
    return normalized_rows


def to_source_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
    elif not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def parse_timestamp(value: object) -> pd.Timestamp | None:
    if not isinstance(value, (str, int, float, date)):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if isinstance(timestamp, pd.Timestamp) else None


def select_eodhd_balance_sheet_row(
    income_date: str, balance_sheet: dict[str, dict]
) -> dict | None:
    row = balance_sheet.get(income_date)
    if isinstance(row, dict):
        return row

    aligned_date = select_closest_date_key(income_date, list(balance_sheet))
    if aligned_date is None:
        return None
    row = balance_sheet.get(aligned_date)
    return row if isinstance(row, dict) else None


def quarterly_start_date(
    market: Literal["t", "u"],
    limit: int,
    quarter_buffer: int,
    month_buffer: int = 0,
) -> str:
    start = pd.Timestamp.now(
        tz=shared.MARKET_TO_TIMEZONE[market]
    ).normalize() - pd.DateOffset(
        months=3 * (limit + quarter_buffer) + month_buffer
    )
    return date_str(start.tz_localize(None))


def token_terminal_headers(slug: str) -> dict[str, str]:
    jwt = os.getenv("TOKEN_TERMINAL_JWT")
    cookie = os.getenv("TOKEN_TERMINAL_COOKIE")
    if not jwt:
        raise ValueError("Token Terminal credentials missing: set TOKEN_TERMINAL_JWT")
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": TOKEN_TERMINAL_AUTHORIZATION,
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": "https://tokenterminal.com",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": (
            f"https://tokenterminal.com/explorer/projects/{slug}/financial-statement"
        ),
        "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", '
        '"Chromium";v="151"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": TOKEN_TERMINAL_USER_AGENT,
        "x-app-path": f"/explorer/projects/{slug}/financial-statement",
        "x-tt-terminal-jwt": jwt,
    }
    if cookie:
        headers["cookie"] = cookie
    return headers


async def fetch_token_terminal_financial_statement(symbol: str) -> list[dict]:
    """Fetch the monthly Token Terminal income-statement metrics for a token."""
    try:
        slug = TOKEN_TERMINAL_PROJECT_SLUGS[symbol.upper()]
    except KeyError as error:
        raise ValueError(
            f"Token Terminal project mapping missing for {symbol}"
        ) from error
    params = {
        "batch": "1",
        "input": json.dumps(
            {
                "0": {"project_slug": slug, "granularity": "month"},
            },
            separators=(",", ":"),
        ),
    }
    text = await shared.cached_get(
        TOKEN_TERMINAL_FINANCIAL_STATEMENT_URL,
        params,
        headers=token_terminal_headers(slug),
    )
    payload = json.loads(text)
    try:
        rows = payload[0]["result"]["data"]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            "Token Terminal financial-statement response is malformed"
        ) from error
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Token Terminal financial-statement response has invalid rows")
    return rows


def build_token_terminal_xps_frame(
    rows: list[dict], periods: int, include_eps: bool
) -> pd.DataFrame:
    """Map Token Terminal rows to dated monthly RPS/EPS inputs."""
    metric_map = {"fees": "rps"}
    if include_eps:
        metric_map["earnings"] = "eps"
    tracked_metrics = [*metric_map, "token_supply_circulating"]
    frame = pd.DataFrame(rows)
    required_columns = {"timestamp", "metric_id", "value"}
    if not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=XPS_COLUMNS, dtype=float)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.loc[frame["metric_id"].isin(tracked_metrics)]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["value"])]
    frame = frame.pivot_table(
        index="timestamp",
        columns="metric_id",
        values="value",
        aggfunc="last",
    ).sort_index()

    incomplete_month_start = (
        pd.Timestamp.now(tz="UTC").normalize().replace(day=1).tz_localize(None)
    )
    frame = frame.loc[frame.index < incomplete_month_start].tail(periods)
    if frame.empty:
        return pd.DataFrame(columns=XPS_COLUMNS, dtype=float)
    expected_dates = pd.date_range(
        end=frame.index[-1], periods=periods, freq="MS"
    )
    frame = frame.reindex(expected_dates)
    supply = frame.get(
        "token_supply_circulating", pd.Series(np.nan, index=frame.index)
    )
    supply = supply.where(supply > 0)
    per_token = frame.reindex(columns=metric_map).div(supply, axis=0)
    per_token = per_token.rename(columns=metric_map).reindex(columns=XPS_COLUMNS)
    per_token.index += pd.DateOffset(months=1)
    return per_token


async def fetch_token_terminal_monthly_xps(
    symbol: str, m: int, include_eps: bool
) -> pd.DataFrame:
    inputs = build_token_terminal_xps_frame(
        await fetch_token_terminal_financial_statement(symbol), m + 11, include_eps
    )
    return complete_and_roll_xps(inputs, 12)


def pivot_finmind_taiwan_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["date", "type", "value"])
    if df.empty:
        return pd.DataFrame()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df.pivot_table(
        index="date",
        columns="type",
        values="value",
        aggfunc="last",
    ).sort_index()


def normalize_finmind_taiwan_income_statement_rows(
    financial_statement_rows: list[dict],
    balance_sheet_rows: list[dict],
) -> dict[str, SourceFieldValues]:
    financial_statement = pivot_finmind_taiwan_rows(financial_statement_rows)
    balance_sheet = pivot_finmind_taiwan_rows(balance_sheet_rows)
    if financial_statement.empty:
        return {}

    df = financial_statement.reindex(columns=["Revenue", "EPS"]).join(
        balance_sheet.reindex(columns=["OrdinaryShare"]), how="left"
    )

    normalized_rows: dict[str, SourceFieldValues] = {}
    for row_date, row in df.iterrows():
        timestamp = parse_timestamp(row_date)
        if timestamp is None:
            continue
        ordinary_shares = row["OrdinaryShare"]
        normalized_row = {
            "revenue": sanitize_source_field_value("revenue", row["Revenue"]),
            "weightedAverageShsOutDil": sanitize_source_field_value(
                "weightedAverageShsOutDil",
                ordinary_shares / FINMIND_TAIWAN_SHARE_PAR_VALUE,
            ),
        }
        normalized_row[EPS_XPS_FIELD] = sanitize_source_field_value(
            EPS_XPS_FIELD, row["EPS"]
        )
        normalized_rows[date_str(timestamp)] = normalized_row
    return normalized_rows


async def fetch_finmind_taiwan_rows(
    dataset: str, symbol: str, start_date: str
) -> list[dict]:
    text = await shared.cached_get(
        FINMIND_API_URL,
        {
            "dataset": dataset,
            "data_id": symbol,
            "start_date": start_date,
        },
        headers={"Authorization": f"Bearer {FINMIND_KEY}"},
    )
    return json.loads(text).get("data", [])


async def fetch_finmind_taiwan_income_statements(
    symbol: str,
    limit: int,
    require_eps: bool = True,
) -> dict[str, dict]:
    start_date = quarterly_start_date(
        "t",
        limit,
        FINMIND_TAIWAN_QUARTER_BUFFER,
        FINMIND_TAIWAN_MONTH_BUFFER,
    )
    financial_statement_rows, balance_sheet_rows = await asyncio.gather(
        fetch_finmind_taiwan_rows("TaiwanStockFinancialStatements", symbol, start_date),
        fetch_finmind_taiwan_rows("TaiwanStockBalanceSheet", symbol, start_date),
    )
    rows = normalize_finmind_taiwan_income_statement_rows(
        financial_statement_rows,
        balance_sheet_rows,
    )
    return select_latest_required_quarters(rows, limit, symbol=symbol)


def build_aligned_source_quarters(
    source_rows: IncomeStatementSources,
) -> AlignedQuarters:
    aligned_quarters = {}
    for source_name in us_income_statement_source_order():
        align_source_rows(
            aligned_quarters, source_name, source_rows.get(source_name, {})
        )
    return dict(sorted(aligned_quarters.items()))


def select_latest_required_quarters[T](
    quarters: dict[str, T],
    limit: int,
    *,
    symbol: str | None = None,
    quarter_label: str = "quarters",
) -> dict[str, T]:
    selected = dict(sorted(quarters.items())[-limit:])
    if len(selected) != limit:
        subject = f" for {symbol}" if symbol is not None else ""
        raise ValueError(
            f"need {limit} {quarter_label}, found {len(selected)}{subject}; "
            f"available={sorted(quarters)}"
        )
    return selected


def required_xps_fields(include_eps: bool) -> list[str]:
    return [*BASE_XPS_FIELDS, *((EPS_XPS_FIELD,) if include_eps else ())]


def date_str(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%d")


def format_sec_cik(cik: object) -> str | None:
    if cik is None:
        return None
    if isinstance(cik, float) and cik.is_integer():
        cik = int(cik)
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(10)


def normalize_fiscal_quarter(value: object) -> SecFiscalQuarter | None:
    if isinstance(value, str):
        value = value.upper().removeprefix("Q")
    match value:
        case 1 | "1":
            return "Q1"
        case 2 | "2":
            return "Q2"
        case 3 | "3":
            return "Q3"
        case 4 | "4":
            return "Q4"
    return None


def dedupe_sec_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "form" not in df:
        return empty_sec_rows()
    df = df.loc[df["form"].isin(SEC_ALLOWED_FORMS)].copy()
    if df.empty:
        logger.warning("SEC reference lookup skipped; no supported forms found")
        return empty_sec_rows()
    missing_columns = [column for column in SEC_DEDUPE_COLS if column not in df]
    if missing_columns:
        logger.warning(
            "SEC reference lookup skipped; missing expected columns {}",
            missing_columns,
        )
        return empty_sec_rows()
    for column in ("filed", "start", "end"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.dropna(subset=SEC_DEDUPE_COLS).sort_values("filed", kind="mergesort")
    period_columns = ["start", "end"]
    amendment_mask = df["form"].str.endswith("/A", na=False)
    first_originals = df[~amendment_mask].drop_duplicates(period_columns, keep="first")
    latest_amendments = df[amendment_mask].drop_duplicates(period_columns, keep="last")
    return (
        pd.concat([first_originals, latest_amendments])
        .drop_duplicates(period_columns, keep="last")
        .loc[:, SEC_DEDUPE_COLS]
    )


def format_sec_candidate_block(matches: pd.DataFrame) -> str:
    lines = ["  SEC candidates:"]
    for _, row in matches.sort_values(["end", "start", "filed"]).iterrows():
        lines.append(
            "    "
            f"filed={date_str(row['filed'])} "
            f"period={date_str(row['start'])}..{date_str(row['end'])} "
            f"value={format_source_log_value(row['val'])}"
        )
    return "\n".join(lines)


def select_sec_fact(
    df: pd.DataFrame,
    description: str,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    *,
    log_errors: bool = True,
) -> pd.Series | None:
    if df.empty:
        return None
    tolerance = pd.DateOffset(months=1)
    min_start = expected_start - tolerance
    max_start = expected_start + tolerance
    min_end = expected_end - tolerance
    max_end = expected_end + tolerance
    matches = df[
        df["start"].between(min_start, max_start, inclusive="neither")
        & df["end"].between(min_end, max_end, inclusive="neither")
    ]
    if matches.empty:
        if log_errors:
            logger.warning(
                "SEC fact lookup returned no matches: {} start=({}, {}) end=({}, {})",
                description,
                date_str(min_start),
                date_str(max_start),
                date_str(min_end),
                date_str(max_end),
            )
        return None
    if len(matches) > 1:
        if log_errors:
            logger.warning(
                "SEC fact lookup returned multiple matches: {} start=({}, {}) end=({}, {}) matches={}\n{}",
                description,
                date_str(min_start),
                date_str(max_start),
                date_str(min_end),
                date_str(max_end),
                len(matches),
                format_sec_candidate_block(matches),
            )
        return None
    return matches.iloc[-1]


def derive_sec_q4_value(
    value_kind: SecQ4ValueKind,
    df: pd.DataFrame,
    description_prefix: str,
    end: pd.Timestamp,
    *,
    log_errors: bool = True,
) -> int | float | None:
    annual = select_sec_fact(
        df,
        description=f"{description_prefix} annual {date_str(end)}",
        expected_start=end - pd.DateOffset(months=12),
        expected_end=end,
        log_errors=log_errors,
    )
    if annual is None:
        return None

    annual_start = annual["start"]
    q1_to_q3 = select_sec_fact(
        df,
        description=f"{description_prefix} Q1-Q3 from {date_str(annual_start)}",
        expected_start=annual_start,
        expected_end=annual_start + pd.DateOffset(months=9),
        log_errors=log_errors,
    )
    if q1_to_q3 is None:
        return None
    if value_kind == "average":
        return round(annual["val"] * 4 - q1_to_q3["val"] * 3)
    return annual["val"] - q1_to_q3["val"]


def lookup_sec_field_value(
    field: str,
    cik: str,
    metadata: SecQuarterMetadata,
    sec_rows: pd.DataFrame,
    *,
    q4_value_kind: SecQ4ValueKind,
    log_errors: bool = False,
) -> int | float | None:
    end = parse_timestamp(metadata.date)
    if end is None:
        return None
    period = metadata.period
    datum = select_sec_fact(
        sec_rows,
        description=f"CIK{cik} {field} {date_str(end)}",
        expected_start=end - pd.DateOffset(months=3),
        expected_end=end,
        log_errors=log_errors and period != "Q4",
    )
    if datum is not None:
        return datum["val"]
    if period != "Q4":
        return None

    return derive_sec_q4_value(
        q4_value_kind,
        sec_rows,
        f"CIK{cik} {field}",
        end,
        log_errors=log_errors,
    )


async def fetch_sec_concept_rows(cik: str, concept: str, unit: str) -> pd.DataFrame:
    text = await shared.cached_get(
        SEC_COMPANY_CONCEPT_URL.format(cik, concept),
        headers={"User-Agent": SEC_USER_AGENT},
    )
    data = json.loads(text)
    unit_rows = data.get("units", {}).get(unit, [])
    if len(unit_rows) == 0:
        logger.warning(
            'SEC concept {} response had empty units["{}"] data for CIK {}; skipping',
            concept,
            unit,
            cik,
        )
        return empty_sec_rows()
    return dedupe_sec_rows(unit_rows)


async def fetch_first_available_sec_field(
    cik: str,
    field: str,
) -> tuple[SecFieldSpec, pd.DataFrame] | None:
    for spec in SEC_FIELD_SPECS.get(field, ()):
        try:
            sec_rows = await fetch_sec_concept_rows(cik, spec.concept, spec.unit)
        except Exception as exc:
            logger.warning(
                "SEC concept {} fetch/processing failed for CIK {}; skipping: {}",
                spec.concept,
                cik,
                exc,
            )
            continue
        if not sec_rows.empty:
            return spec, sec_rows
    return None


async def fetch_fmp_income_statements(
    market: Literal["j", "t", "u"],
    symbol: str,
    limit: int,
) -> dict[str, dict]:
    params = {
        "apikey": FMP_KEY,
        "limit": limit,
        "period": "quarter",
    }
    if market == "u":
        url = "https://financialmodelingprep.com/stable/income-statement"
        params["symbol"] = symbol
        eps_field = "epsDiluted"
    else:
        url = (
            f"https://financialmodelingprep.com/api/v3/income-statement/"
            f"{add_suffix(market, symbol)}"
        )
        eps_field = "epsdiluted"
    text = await shared.cached_get(url, params)
    return normalize_income_statement_rows(
        json.loads(text),
        field_map=xps_field_map("weightedAverageShsOutDil", eps_field),
        date_field="date",
        metadata_fields={
            "cik": "cik",
            "quarter": "period",
        },
    )


async def fetch_massive_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        "tickers": symbol,
        "timeframe": "quarterly",
        "limit": limit,
        "sort": "period_end.desc",
        "apiKey": MASSIVE_API_KEY,
    }
    text = await shared.cached_get(
        "https://api.massive.com/stocks/financials/v1/income-statements", params
    )
    return normalize_income_statement_rows(
        json.loads(text)["results"] or [],
        field_map=xps_field_map(
            "diluted_shares_outstanding", "diluted_earnings_per_share"
        ),
        date_field="period_end",
        metadata_fields={
            "cik": "cik",
            "quarter": "fiscal_quarter",
        },
    )


async def fetch_finnhub_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        "symbol": symbol,
        "statement": "ic",
        "freq": "quarterly",
        "preliminary": "false",
        "token": FINNHUB_API_KEY,
    }
    text = await shared.cached_get("https://finnhub.io/api/v1/stock/financials", params)

    financials = sorted(
        json.loads(text)["financials"] or [],
        key=lambda row: row["period"],
        reverse=True,
    )[:limit]
    return normalize_income_statement_rows(
        financials,
        field_map=xps_field_map("dilutedAverageSharesOutstanding", "dilutedEPS"),
        date_field="period",
        base_field_scale=FINNHUB_MILLION_SCALE,
    )


def normalize_tiingo_income_statement_rows(statements: list[dict]) -> dict[str, dict]:
    """Normalize Tiingo's nested quarterly income-statement response."""
    rows = {}
    field_map = xps_field_map("shareswaDil", "epsDil")
    for statement in statements:
        if normalize_fiscal_quarter(statement.get("quarter")) is None:
            continue
        statement_data = statement.get("statementData", {})
        if not isinstance(statement_data, dict):
            continue
        income_statement = statement_data.get("incomeStatement", [])
        if not isinstance(income_statement, list) or "date" not in statement:
            continue
        values = {
            entry.get("dataCode"): entry.get("value")
            for entry in income_statement
            if isinstance(entry, dict) and entry.get("dataCode") is not None
        }
        date = parse_timestamp(statement["date"])
        if date is not None:
            rows[date_str(date)] = normalize_source_row(values, field_map)
    return rows


async def fetch_tiingo_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        # Use Tiingo's latest restated values, whose dates are fiscal period ends.
        "asReported": "false",
        "startDate": quarterly_start_date("u", limit, TIINGO_QUARTER_BUFFER),
    }
    text = await shared.cached_get(
        TIINGO_FUNDAMENTALS_STATEMENTS_URL.format(symbol),
        params,
        headers={"Authorization": f"Token {TIINGO_API_KEY}"},
    )
    return normalize_tiingo_income_statement_rows(json.loads(text))


def diluted_eps_from_reported_income(
    income_statement_row: dict, diluted_shares: int | float | None
) -> float | None:
    if not diluted_shares:
        return None
    net_income_applicable = to_source_number(
        income_statement_row.get("netIncomeApplicableToCommonShares")
    )
    if net_income_applicable is not None:
        earnings = net_income_applicable
    else:
        net_income = to_source_number(income_statement_row.get("netIncome"))
        preferred_adjustments = to_source_number(
            income_statement_row.get("preferredStockAndOtherAdjustments")
        )
        if preferred_adjustments is not None and net_income is not None:
            earnings = net_income - preferred_adjustments
        else:
            earnings = net_income
    if earnings is None:
        return None
    return round(earnings / diluted_shares, 6)


async def fetch_eodhd_fundamentals(symbol: str) -> dict:
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "filter": (
            "Financials::Income_Statement::quarterly,"
            "Financials::Balance_Sheet::quarterly"
        ),
    }
    text = await shared.cached_get(
        EODHD_FUNDAMENTALS_URL.format(f"{symbol}.US"), params
    )
    return json.loads(text)


async def fetch_eodhd_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    data = await fetch_eodhd_fundamentals(symbol)
    income_statement = data.get("Financials::Income_Statement::quarterly", {})
    balance_sheet = data.get("Financials::Balance_Sheet::quarterly", {})

    rows = {}
    for income_date, income_row in sorted(income_statement.items(), reverse=True):
        if not isinstance(income_row, dict):
            continue

        balance_sheet_row = (
            select_eodhd_balance_sheet_row(income_date, balance_sheet) or {}
        )
        diluted_shares = to_source_number(
            balance_sheet_row.get("commonStockSharesOutstanding")
        )
        normalized_row = {
            "revenue": sanitize_source_field_value(
                "revenue", to_source_number(income_row.get("totalRevenue"))
            ),
            "weightedAverageShsOutDil": sanitize_source_field_value(
                "weightedAverageShsOutDil", diluted_shares
            ),
        }
        normalized_row[EPS_XPS_FIELD] = sanitize_source_field_value(
            EPS_XPS_FIELD,
            diluted_eps_from_reported_income(income_row, diluted_shares),
        )
        rows[income_date] = normalized_row
        if len(rows) >= limit:
            break
    return rows


def field_has_consensus(field: str, lhs: int | float, rhs: int | float) -> bool:
    if field == EPS_XPS_FIELD and abs(float(lhs) - float(rhs)) < EPS_ABS_TOLERANCE:
        return True
    return source_log_diff(lhs, rhs) < SOURCE_DIFF_THRESHOLD


def usable_source_values(
    source_values: dict[str, int | float | None],
    *,
    include_sec: bool = True,
) -> list[tuple[str, int | float]]:
    source_names = us_income_statement_source_order(include_sec=include_sec)
    values: list[tuple[str, int | float]] = []
    for source_name in source_names:
        value = source_values.get(source_name)
        if not has_source_field_value(value):
            continue
        values.append((source_name, value))
    return values


def build_connected_consensus_groups(
    field: str,
    values: list[tuple[str, int | float]],
) -> list[tuple[tuple[str, int | float], ...]]:
    adjacent_sources = {source_name: set() for source_name, _ in values}
    for (lhs_name, lhs_value), (rhs_name, rhs_value) in combinations(values, 2):
        if not field_has_consensus(field, lhs_value, rhs_value):
            continue
        adjacent_sources[lhs_name].add(rhs_name)
        adjacent_sources[rhs_name].add(lhs_name)

    consensus_groups = []
    visited_sources = set()
    for source_name, _ in values:
        if source_name in visited_sources or not adjacent_sources[source_name]:
            continue

        group_names = set()
        pending_sources = [source_name]
        visited_sources.add(source_name)
        while pending_sources:
            current_source = pending_sources.pop()
            group_names.add(current_source)
            for adjacent_source in adjacent_sources[current_source]:
                if adjacent_source in visited_sources:
                    continue
                visited_sources.add(adjacent_source)
                pending_sources.append(adjacent_source)

        consensus_groups.append(
            tuple(item for item in values if item[0] in group_names)
        )
    return consensus_groups


def choose_consensus_group(
    groups: list[tuple[tuple[str, int | float], ...]],
) -> tuple[tuple[str, int | float], ...] | None:
    if not groups:
        return None

    largest_group_size = max(len(group) for group in groups)
    largest_groups = [group for group in groups if len(group) == largest_group_size]
    if len(largest_groups) == 1:
        return largest_groups[0]
    return None


def resolve_source_consensus(
    field: str,
    source_values: dict[str, int | float | None],
) -> Consensus | None:
    values = usable_source_values(source_values)
    consensus_group = choose_consensus_group(
        build_connected_consensus_groups(field, values)
    )
    if consensus_group is None:
        return None

    return Consensus(
        value=sum(value for _, value in consensus_group) / len(consensus_group),
        sources=tuple(name for name, _ in consensus_group),
    )


def format_source_log_value(value: int | float | None) -> str:
    if not has_source_field_value(value):
        return "missing"
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def format_source_field_block(
    field: str,
    source_values: dict[str, int | float | None],
    quarter_key: str,
) -> str:
    lines = [f"  {field}:"]
    for source_name in us_income_statement_source_order(include_sec=True):
        lines.append(
            "    "
            f"{SOURCE_LABELS[source_name]:<8} "
            f"date={quarter_key:<10} "
            f"value={format_source_log_value(source_values.get(source_name))}"
        )
    return "\n".join(lines)


def format_source_fetch_error(error: Exception) -> str:
    """Summarize provider failures without logging request URLs or API keys."""
    if isinstance(error, HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


async def fetch_us_income_statement_source(
    source: USIncomeStatementSource,
    symbol: str,
    limit: int,
) -> tuple[str, SourceIncomeStatementRows | None]:
    try:
        return source.name, await source.fetcher(symbol, limit)
    except (HTTPError, json.JSONDecodeError) as error:
        logger.warning(
            "{} income statements unavailable for {}: {}; skipping source",
            source.label,
            symbol,
            format_source_fetch_error(error),
        )
        return source.name, None
    except Exception:
        logger.exception(
            "{} income statement fetch failed unexpectedly for {}; skipping source",
            source.label,
            symbol,
        )
        return source.name, None


async def fetch_us_income_statement_sources(
    symbol: str, limit: int
) -> IncomeStatementSources:
    results = await asyncio.gather(
        *(
            fetch_us_income_statement_source(source, symbol, limit)
            for source in ENABLED_US_INCOME_STATEMENT_SOURCES
        )
    )
    return {name: rows if rows is not None else {} for name, rows in results}


def iter_sec_metadata_periods(
    aligned_quarters: AlignedQuarters,
) -> Iterator[tuple[str, list[SourcePeriod]]]:
    for source_name in SEC_METADATA_SOURCES:
        for quarter_key, quarter in aligned_quarters.items():
            source_periods = quarter.source_periods.get(source_name, [])
            if source_periods:
                yield quarter_key, source_periods


def select_sec_cik(aligned_quarters: AlignedQuarters) -> str | None:
    # TODO: Revisit CIK source/date ordering. This currently prefers any FMP CIK
    # over Massive, independently of the source row chosen for quarter metadata.
    for _, source_periods in iter_sec_metadata_periods(aligned_quarters):
        for source_period in source_periods:
            cik = format_sec_cik(source_period.row.get("cik"))
            if cik is not None:
                return cik
    return None


def build_sec_quarter_metadata(
    aligned_quarters: AlignedQuarters,
) -> dict[str, SecQuarterMetadata]:
    metadata = {}
    for quarter_key, source_periods in iter_sec_metadata_periods(aligned_quarters):
        if quarter_key in metadata:
            continue
        source_period = source_periods[0]
        period = normalize_fiscal_quarter(source_period.row.get("quarter"))
        if period is not None:
            metadata[quarter_key] = SecQuarterMetadata(
                date=source_period.date,
                period=period,
            )
    return metadata


async def fetch_sec_values_for_quarters(
    cik: str,
    metadata_by_quarter: dict[str, SecQuarterMetadata],
    fields: list[str],
) -> dict[str, dict[str, int | float]]:
    """Fetch usable SEC field values keyed by aligned quarter."""

    async def fetch_field_values(field_name: str) -> dict[str, int | float]:
        available = await fetch_first_available_sec_field(cik, field_name)
        if available is None:
            return {}
        spec, sec_rows = available

        values = {}
        for quarter_key, metadata in metadata_by_quarter.items():
            value = lookup_sec_field_value(
                field_name,
                cik,
                metadata,
                sec_rows,
                q4_value_kind=spec.q4_value_kind,
            )
            value = sanitize_source_field_value(field_name, value)
            if value is not None:
                values[quarter_key] = value
        return values

    values_by_quarter = {}
    field_names = dict.fromkeys(fields)
    field_values = await asyncio.gather(
        *(fetch_field_values(field_name) for field_name in field_names)
    )
    for field_name, values in zip(field_names, field_values, strict=True):
        for quarter_key, value in values.items():
            values_by_quarter.setdefault(quarter_key, {})[field_name] = value
    return values_by_quarter


async def add_sec_reference_values(
    symbol: str,
    aligned_quarters: AlignedQuarters,
    fields: list[str],
) -> None:
    """Add usable SEC facts to aligned US quarters as reference-source values."""
    cik = select_sec_cik(aligned_quarters)
    if cik is None:
        logger.warning("SEC reference unavailable for {}: no FMP/Massive CIK", symbol)
        return

    metadata_by_quarter = build_sec_quarter_metadata(aligned_quarters)
    if not metadata_by_quarter:
        logger.warning(
            "SEC reference unavailable for {}: no FMP/Massive quarter metadata",
            symbol,
        )
        return

    values_by_quarter = await fetch_sec_values_for_quarters(
        cik,
        metadata_by_quarter,
        fields,
    )
    for quarter_key, field_values in values_by_quarter.items():
        quarter = aligned_quarters[quarter_key]
        for field_name, value in field_values.items():
            quarter.fields.setdefault(field_name, {})["sec"] = value


def resolve_us_quarter_consensus(
    symbol: str,
    anchor_date: str,
    quarter: AlignedQuarter,
    required_fields: list[str],
) -> ResolvedQuarter:
    resolved_quarter = {}
    for field_name in required_fields:
        source_values = quarter.fields.get(field_name, {})
        consensus = resolve_source_consensus(field_name, source_values)
        if consensus is not None:
            resolved_quarter[field_name] = consensus.value
            continue
        logger.warning(
            "No consensus for {} quarter {} field={}\n{}",
            symbol,
            anchor_date,
            field_name,
            format_source_field_block(field_name, source_values, anchor_date),
        )
        resolved_quarter[field_name] = None
    return resolved_quarter


async def resolve_us_income_statement_quarters(
    symbol: str,
    source_rows: IncomeStatementSources,
    limit: int,
    include_eps: bool,
) -> ResolvedQuarters:
    """Align US provider data, add SEC references, then resolve consensus values."""
    required_fields = required_xps_fields(include_eps)

    # 1. Group provider rows that describe the same fiscal quarter.
    aligned_quarters = build_aligned_source_quarters(source_rows)

    # 2. Add SEC revenue/shares/EPS reference values before validation.
    if shared.ENABLE_SEC_REFERENCE:
        await add_sec_reference_values(
            symbol,
            aligned_quarters,
            required_fields,
        )

    # 3. Retain the requested window. Metric completeness is checked centrally.
    selected_quarters = select_latest_required_quarters(
        aligned_quarters, limit, symbol=symbol, quarter_label="aligned quarters"
    )

    # 4. Optionally record diagnostics for the same quarters that will be returned.
    if include_eps and shared.ENABLE_XPS_DIAGNOSTICS:
        record_xps_diagnostics(symbol, selected_quarters)

    # 5. Reduce each field to the mean of its winning consensus group.
    resolved_quarters = {
        anchor_date: resolve_us_quarter_consensus(
            symbol, anchor_date, quarter, required_fields
        )
        for anchor_date, quarter in selected_quarters.items()
    }

    return resolved_quarters


async def fetch_resolved_income_statement_quarters(
    market: Literal["j", "t", "u"],
    symbol: str,
    limit: int,
    include_eps: bool,
) -> ResolvedQuarters:
    if market == "t":
        return await fetch_finmind_taiwan_income_statements(
            symbol, limit, require_eps=include_eps
        )

    if market != "u":
        fmp_rows = await fetch_fmp_income_statements(market, symbol, limit)
        return select_latest_required_quarters(fmp_rows, limit, symbol=symbol)

    source_rows = await fetch_us_income_statement_sources(symbol, limit + 1)
    return await resolve_us_income_statement_quarters(
        symbol,
        source_rows,
        limit,
        include_eps,
    )


def build_quarterly_xps_inputs(
    resolved_quarters: ResolvedQuarters, include_eps: bool
) -> pd.DataFrame:
    """Derive dated quarterly RPS/EPS inputs without validating completeness."""
    df = pd.DataFrame.from_dict(resolved_quarters, orient="index")
    df.index = pd.to_datetime(df.index) + pd.Timedelta(days=1)
    df = df.reindex(columns=required_xps_fields(include_eps))
    inputs = pd.DataFrame(index=df.index)
    inputs["rps"] = df["revenue"] / df["weightedAverageShsOutDil"]
    inputs["eps"] = df[EPS_XPS_FIELD] if include_eps else np.nan
    return inputs.reindex(columns=XPS_COLUMNS)


def complete_and_roll_xps(xps: pd.DataFrame, window: int) -> pd.DataFrame:
    """Keep complete RPS/EPS columns, then apply the existing rolling TTM logic."""
    complete = (
        xps.reindex(columns=XPS_COLUMNS)
        .sort_index()
        .dropna(axis="columns", how="any")
    )
    if complete.empty:
        raise ValueError("neither a complete RPS nor EPS series is available")
    return complete.rolling(window).sum().iloc[window - 1 :]


def build_xps_frame(
    resolved_quarters: ResolvedQuarters, include_eps: bool
) -> pd.DataFrame:
    """Build quarterly inputs, then use the shared completeness/TTM path."""
    return complete_and_roll_xps(
        build_quarterly_xps_inputs(resolved_quarters, include_eps), 4
    )


async def fetch_xps(
    market: Literal["c", "j", "t", "u"],
    symbol: str,
    q: int,
    include_eps: bool = True,
) -> pd.DataFrame:
    if market == "c":
        monthly_frame = await fetch_token_terminal_monthly_xps(
            symbol, q * 3, include_eps
        )
        return monthly_frame.iloc[2::3]

    limit = q + 3
    resolved_quarters = await fetch_resolved_income_statement_quarters(
        market, symbol, limit, include_eps
    )
    return build_xps_frame(resolved_quarters, include_eps)


async def calc_px(
    market: Literal["c", "j", "t", "u"], symbol: str, q: int
) -> pd.DataFrame:
    xps_task = (
        fetch_token_terminal_monthly_xps(symbol, q * 3 + 1, True)
        if market == "c"
        else fetch_xps(market, symbol, q + EXTRA_Q)
    )
    prices, xps = await asyncio.gather(
        shared.get_prices(market, symbol, 91 * q, False),
        xps_task,
    )
    xps = xps.reindex(columns=["rps", "eps"])
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(prices),
    )
    df = (
        pd.DataFrame({"price": prices}, index)
        .join(xps, how="outer")
        .ffill()
        .tail(len(index))
    )
    if pd.isna(df["price"].iloc[0]) or df[["rps", "eps"]].iloc[0].isna().all():
        raise ValueError
    df["ps"] = df["price"] / df["rps"]
    df["pe"] = df["price"] / df["eps"].where(df["eps"] > 0)
    return df[["ps", "pe"]]


# def avg_by_period_ends(values, ends):
#     today = pd.Timestamp.now(tz='Asia/Taipei').normalize().tz_localize(None)
#     series = pd.Series(
#         values, index=pd.date_range(end=today, periods=len(values), freq='D')
#     )
#     ends = pd.to_datetime(ends)
#     starts = ends[:-1] + pd.Timedelta(days=1)
#     return pd.Series({e: series.loc[s:e].mean() for s, e in zip(starts, ends[1:])})


# async def calc_rps_from_finmind(symbol: str):
#     url = 'https://api.finmindtrade.com/api/v4/data'
#     params = {
#         'data_id': symbol,
#         'dataset': 'TaiwanStockFinancialStatements',
#         'start_date': arrow.now('Asia/Taipei')
#         .shift(days=-10 * 91)
#         .format('YYYY-MM-DD'),
#     }
#     headers = {'Authorization': f'Bearer {FINMIND_KEY}'}
#     async with AsyncClient() as client:
#         resp, fx = await asyncio.gather(
#             client.get(url, params=params, headers=headers), get_rates(client, 10 * 91)
#         )
#     df = (
#         pd.DataFrame(resp.json()['data'])
#         .pivot(index='date', columns='type', values='value')
#         .sort_index()
#         .iloc[-9:]
#     )
#     if len(df) != 9:
#         raise AssertionError
#     fx = avg_by_period_ends(fx, df.index)
#     df = df.iloc[1:]
#     df['Revenue'] /= fx.values
#     df['rps'] = (
#         (df['Revenue'] * df['EPS'] / df['EquityAttributableToOwnersOfParent'])
#         .rolling(4)
#         .sum()
#     )
#     return df['rps'].iloc[3:]
