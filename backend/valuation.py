import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import pandas as pd
from httpx import HTTPStatusError
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

type XpsValue = int | float | None
type SourceFieldValues = dict[str, XpsValue]


@dataclass
class AlignedQuarter:
    """Aligned field values plus source-period metadata for SEC lookup."""

    fields: dict[str, SourceFieldValues] = field(default_factory=dict)
    source_periods: dict[str, list[tuple[str, IncomeStatementRow]]] = field(
        default_factory=dict
    )


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
    fetch: Callable[[str, int], Awaitable[SourceIncomeStatementRows]]


US_INCOME_STATEMENT_SOURCES = (
    USIncomeStatementSource(
        'fmp',
        'FMP',
        True,
        lambda symbol, limit: fetch_fmp_income_statements('u', symbol, limit),
    ),
    USIncomeStatementSource(
        'massive',
        'Massive',
        shared.ENABLE_MASSIVE_FUNDAMENTALS,
        lambda symbol, limit: fetch_massive_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        'eodhd',
        'EODHD',
        shared.ENABLE_EODHD_FUNDAMENTALS,
        lambda symbol, limit: fetch_eodhd_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        'finnhub',
        'Finnhub',
        True,
        lambda symbol, limit: fetch_finnhub_income_statements(symbol, limit),
    ),
    USIncomeStatementSource(
        'tiingo',
        'Tiingo',
        shared.ENABLE_TIINGO_FUNDAMENTALS,
        lambda symbol, limit: fetch_tiingo_income_statements(symbol, limit),
    ),
)
ENABLED_US_INCOME_STATEMENT_SOURCES = tuple(
    source for source in US_INCOME_STATEMENT_SOURCES if source.enabled
)
SOURCE_LABELS = {
    **{source.name: source.label for source in US_INCOME_STATEMENT_SOURCES},
    'sec': 'SEC',
}


def us_income_statement_source_order(
    include_sec: bool = False,
) -> tuple[str, ...]:
    names = tuple(source.name for source in ENABLED_US_INCOME_STATEMENT_SOURCES)
    return (*names, 'sec') if include_sec else names


BASE_XPS_FIELDS = ('revenue', 'weightedAverageShsOutDil')
EPS_XPS_FIELD = 'epsDiluted'
ALL_XPS_FIELDS = (*BASE_XPS_FIELDS, EPS_XPS_FIELD)


def new_xps_missing_counts() -> dict[str, dict[str, int]]:
    return {
        field: {source: 0 for source in us_income_statement_source_order()}
        for field in ALL_XPS_FIELDS
    }


def new_xps_consensus_pair_counts() -> dict[str, dict[str, int]]:
    source_names = us_income_statement_source_order()
    return {
        field: {
            f'{left}:{right}': 0
            for index, left in enumerate(source_names)
            for right in source_names[index + 1 :]
        }
        for field in ALL_XPS_FIELDS
    }


@dataclass
class XpsDiagnostics:
    seen: set[tuple[str, str]]
    missing: dict[str, dict[str, int]]
    consensus_pairs: dict[str, dict[str, int]]


def new_xps_diagnostics() -> XpsDiagnostics:
    return XpsDiagnostics(
        seen=set(),
        missing=new_xps_missing_counts(),
        consensus_pairs=new_xps_consensus_pair_counts(),
    )


# Diagnostics are intentionally process-local. Restarting the backend starts a new
# screener run with clean counts. SEC is a reference source, not a peer
# provider, so it is excluded.
_xps_diagnostics = new_xps_diagnostics()
SEC_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
)
SEC_COMPANY_CONCEPT_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'
)
FINMIND_API_URL = 'https://api.finmindtrade.com/api/v4/data'
EODHD_FUNDAMENTALS_URL = 'https://eodhd.com/api/v1.1/fundamentals/{}'
TIINGO_FUNDAMENTALS_STATEMENTS_URL = (
    'https://api.tiingo.com/tiingo/fundamentals/{}/statements'
)
FINMIND_TAIWAN_QUARTER_BUFFER = 2
FMP_TAIWAN_QUARTER_BUFFER = 1
TIINGO_QUARTER_BUFFER = 1
FINMIND_TAIWAN_SHARE_PAR_VALUE = 10
SEC_DEDUPE_COLS = ['filed', 'val', 'start', 'end']
SEC_ALLOWED_FORMS = {'10-Q', '10-K', '10-Q/A', '10-K/A'}
SEC_REFERENCE_FIELDS = ('weightedAverageShsOutDil', EPS_XPS_FIELD)
SEC_METADATA_SOURCES = ('fmp', 'massive')
SEC_FIELD_CONCEPTS = {
    'weightedAverageShsOutDil': (
        ('WeightedAverageNumberOfDilutedSharesOutstanding', 'shares'),
    ),
    EPS_XPS_FIELD: (('EarningsPerShareDiluted', 'USD/shares'),),
}


@dataclass(frozen=True)
class IncomeStatementFetch:
    """Fetched source rows together with provider-level availability metadata."""

    rows: IncomeStatementSources
    unavailable_sources: frozenset[str]


def empty_sec_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=SEC_DEDUPE_COLS)


def source_log_diff(lhs: int | float, rhs: int | float) -> float:
    lhs = float(lhs)
    rhs = float(rhs)
    if not all(map(math.isfinite, (lhs, rhs))):
        return math.inf
    if lhs == rhs:
        return 0.0
    if lhs == 0 or rhs == 0:
        return math.inf
    ratio = lhs / rhs
    if ratio <= 0:
        return math.inf
    return abs(math.log(ratio))


def source_abs_diff(lhs: int | float, rhs: int | float) -> float:
    lhs = float(lhs)
    rhs = float(rhs)
    if not all(map(math.isfinite, (lhs, rhs))):
        return math.inf
    return abs(lhs - rhs)


def has_source_field_value(value: int | float | None) -> bool:
    return value is not None and not pd.isna(value)


def missing_xps_sources(
    quarter: AlignedQuarter,
    field: str,
    unavailable_sources: frozenset[str],
) -> list[str]:
    return [
        source
        for source in us_income_statement_source_order()
        if source not in unavailable_sources
        and not has_source_field_value(quarter.fields.get(field, {}).get(source))
    ]


def record_xps_diagnostics(
    symbol: str,
    aligned_quarters: AlignedQuarters,
    *,
    unavailable_sources: frozenset[str] = frozenset(),
) -> None:
    """Record missing values and pairwise consensus for selected quarters."""
    diagnostics = _xps_diagnostics
    for quarter, data in aligned_quarters.items():
        key = (symbol, quarter)
        if key in diagnostics.seen:
            continue

        diagnostics.seen.add(key)
        for field in ALL_XPS_FIELDS:
            counts = diagnostics.missing[field]
            for source in missing_xps_sources(data, field, unavailable_sources):
                counts[source] += 1

            source_values = data.fields.get(field, {})
            peer_values = usable_source_values(
                source_values, us_income_statement_source_order()
            )
            pair_counts = diagnostics.consensus_pairs[field]
            for (left_source, left_value), (right_source, right_value) in combinations(
                peer_values, 2
            ):
                if field_has_consensus(field, left_value, right_value):
                    pair_counts[f'{left_source}:{right_source}'] += 1


def get_xps_diagnostics() -> dict:
    """Return missing-value and pairwise-consensus counts."""
    return {
        'total_quarters': len(_xps_diagnostics.seen),
        'missing': _xps_diagnostics.missing,
        'consensus_pairs': _xps_diagnostics.consensus_pairs,
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

    matches.sort(key=lambda item: (item[0], item[1]))
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
        quarter.source_periods.setdefault(source_name, []).append((source_date, row))
        for field in ALL_XPS_FIELDS:
            if field not in row:
                continue
            quarter.fields.setdefault(field, {})[source_name] = row[field]


def sanitize_source_field_value(
    field: str, value: int | float | None
) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if field in {'revenue', 'weightedAverageShsOutDil'} and value <= 0:
        return None
    return value


def normalize_source_row(
    row: dict,
    field_map: dict[str, str],
    value_transform=lambda field, value: value,
) -> dict[str, int | float | None]:
    return {
        field: sanitize_source_field_value(
            field,
            value_transform(field, row.get(source_field)),
        )
        for field, source_field in field_map.items()
    }


def normalize_income_statement_rows(
    rows: list[dict],
    *,
    field_map: dict[str, str],
    date_field: str,
    metadata_fields: dict[str, str] | None = None,
    value_transform=lambda field, value: value,
) -> dict[str, dict]:
    normalized_rows = {}
    for row in rows:
        normalized_row = normalize_source_row(
            row,
            field_map,
            value_transform=value_transform,
        )
        if metadata_fields is not None:
            normalized_row.update(
                {
                    output_field: row.get(source_field)
                    for output_field, source_field in metadata_fields.items()
                }
            )
        normalized_rows[row[date_field]] = normalized_row
    return normalized_rows


def finnhub_million(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    return value * FINNHUB_MILLION_SCALE


def to_source_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def normalize_finnhub_field_value(
    field: str, value: int | float | None
) -> int | float | None:
    if field in BASE_XPS_FIELDS:
        return finnhub_million(value)
    return value


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


def finmind_taiwan_start_date(
    limit: int, quarter_buffer: int = FINMIND_TAIWAN_QUARTER_BUFFER
) -> str:
    quarter_count = limit + quarter_buffer
    start = pd.Timestamp.now(
        tz=shared.MARKET_TO_TIMEZONE['t']
    ).normalize() - pd.DateOffset(months=3 * quarter_count)
    return date_str(start.tz_localize(None))


def pivot_finmind_taiwan_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=['date', 'type', 'value'])
    if df.empty:
        return pd.DataFrame()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    return df.pivot_table(
        index='date',
        columns='type',
        values='value',
        aggfunc='last',
    ).sort_index()


def normalize_finmind_taiwan_income_statement_rows(
    financial_statement_rows: list[dict],
    balance_sheet_rows: list[dict],
) -> dict[str, dict]:
    financial_statement = pivot_finmind_taiwan_rows(financial_statement_rows)
    balance_sheet = pivot_finmind_taiwan_rows(balance_sheet_rows)
    if financial_statement.empty:
        return {}

    required_financial_statement_fields = ['Revenue']
    missing_financial_statement_fields = [
        field
        for field in required_financial_statement_fields
        if field not in financial_statement
    ]
    if missing_financial_statement_fields:
        logger.warning(
            'FinMind Taiwan financial statement data missing fields {}; skipping rows',
            missing_financial_statement_fields,
        )
        return {}
    if 'OrdinaryShare' not in balance_sheet:
        logger.warning(
            'FinMind Taiwan balance sheet data missing OrdinaryShare; '
            'leaving share counts empty'
        )
        balance_sheet = pd.DataFrame(index=financial_statement.index)
        balance_sheet['OrdinaryShare'] = None

    df = financial_statement.reindex(columns=['Revenue', 'EPS']).join(
        balance_sheet[['OrdinaryShare']], how='outer'
    )
    df = df.dropna(subset=required_financial_statement_fields, how='all')

    normalized_rows = {}
    for date, row in df.iterrows():
        ordinary_shares = row['OrdinaryShare']
        normalized_row = {
            'revenue': sanitize_source_field_value('revenue', row['Revenue']),
            'weightedAverageShsOutDil': sanitize_source_field_value(
                'weightedAverageShsOutDil',
                (
                    ordinary_shares / FINMIND_TAIWAN_SHARE_PAR_VALUE
                    if pd.notna(ordinary_shares)
                    else None
                ),
            ),
        }
        normalized_row[EPS_XPS_FIELD] = sanitize_source_field_value(
            EPS_XPS_FIELD, row['EPS']
        )
        normalized_rows[date] = normalized_row
    return normalized_rows


def merge_preferred_xps_rows(
    baseline_rows: dict[str, dict],
    preferred_rows: dict[str, dict],
) -> None:
    for preferred_date, preferred_row in preferred_rows.items():
        baseline_date = select_closest_date_key(preferred_date, list(baseline_rows))
        if baseline_date is None:
            continue
        baseline_row = baseline_rows[baseline_date]
        for field in ALL_XPS_FIELDS:
            value = preferred_row.get(field)
            if has_source_field_value(value):
                baseline_row[field] = value


def row_has_required_xps_fields(row: dict, required_fields: list[str]) -> bool:
    return all(has_source_field_value(row.get(field)) for field in required_fields)


def drop_incomplete_latest_quarter(
    rows: dict[str, dict], required_fields: list[str]
) -> dict[str, dict]:
    dates = sorted(rows)
    if not dates:
        return rows
    latest_date = dates[-1]
    if row_has_required_xps_fields(rows[latest_date], required_fields):
        return rows
    return {date: rows[date] for date in dates[:-1]}


def select_latest_clean_required_quarters(
    rows: dict[str, dict],
    limit: int,
    required_fields: list[str],
    *,
    symbol: str | None = None,
) -> dict[str, dict]:
    selected = select_latest_required_quarters(rows, limit, symbol=symbol)
    dirty_dates = [
        date
        for date, row in selected.items()
        if not row_has_required_xps_fields(row, required_fields)
    ]
    if dirty_dates:
        raise ValueError(f'incomplete quarters: {dirty_dates}')
    return selected


async def fetch_finmind_taiwan_rows(
    dataset: str, symbol: str, start_date: str
) -> list[dict]:
    text = await shared.cached_get(
        FINMIND_API_URL,
        {
            'dataset': dataset,
            'data_id': symbol,
            'start_date': start_date,
            'token': FINMIND_KEY,
        },
    )
    return json.loads(text).get('data', [])


async def fetch_finmind_taiwan_income_statements(
    symbol: str,
    limit: int,
    require_eps: bool = True,
) -> dict[str, dict]:
    start_date = finmind_taiwan_start_date(limit)
    required_fields = required_xps_fields(require_eps)
    financial_statement_rows, balance_sheet_rows, rows = await asyncio.gather(
        fetch_finmind_taiwan_rows('TaiwanStockFinancialStatements', symbol, start_date),
        fetch_finmind_taiwan_rows('TaiwanStockBalanceSheet', symbol, start_date),
        fetch_fmp_income_statements(
            't',
            symbol,
            limit + FMP_TAIWAN_QUARTER_BUFFER,
        ),
    )
    finmind_rows = normalize_finmind_taiwan_income_statement_rows(
        financial_statement_rows,
        balance_sheet_rows,
    )
    merge_preferred_xps_rows(rows, finmind_rows)
    rows = drop_incomplete_latest_quarter(rows, required_fields)
    return select_latest_clean_required_quarters(
        rows, limit, required_fields, symbol=symbol
    )


def build_aligned_source_quarters(
    source_rows: IncomeStatementSources,
) -> AlignedQuarters:
    aligned_quarters = {}
    for source_name in us_income_statement_source_order():
        align_source_rows(
            aligned_quarters, source_name, source_rows.get(source_name, {})
        )
    return dict(sorted(aligned_quarters.items()))


def select_latest_required_quarters(
    quarters: dict[str, dict],
    limit: int,
    *,
    symbol: str | None = None,
    quarter_label: str = 'quarters',
) -> dict[str, dict]:
    selected = dict(sorted(quarters.items())[-limit:])
    if len(selected) != limit:
        subject = f' for {symbol}' if symbol is not None else ''
        raise ValueError(
            f'need {limit} {quarter_label}, found {len(selected)}{subject}; '
            f'available={sorted(quarters)}'
        )
    return selected


def required_xps_fields(include_eps: bool) -> list[str]:
    return [*BASE_XPS_FIELDS, *((EPS_XPS_FIELD,) if include_eps else ())]


def date_str(date: pd.Timestamp) -> str:
    return date.strftime('%Y-%m-%d')


def quarter_end_bounds(end: pd.Timestamp, months: int = 1) -> tuple[str, str]:
    return (
        date_str(end - pd.DateOffset(months=months)),
        date_str(end + pd.DateOffset(months=months)),
    )


def format_sec_cik(cik: int | str | None) -> str | None:
    if cik is None:
        return None
    if isinstance(cik, float) and cik.is_integer():
        cik = int(cik)
    digits = ''.join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(10)


def normalize_massive_fiscal_quarter(value: object) -> str | None:
    if value is None:
        return None
    try:
        quarter = int(value)
    except (TypeError, ValueError):
        value = str(value).upper()
        return value if value in {'Q1', 'Q2', 'Q3', 'Q4'} else None
    if quarter not in {1, 2, 3, 4}:
        return None
    return f'Q{quarter}'


def dedupe_sec_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or 'form' not in df:
        return empty_sec_rows()
    df = df[df['form'].isin(SEC_ALLOWED_FORMS)]
    if df.empty:
        logger.warning('SEC reference lookup skipped; no supported forms found')
        return empty_sec_rows()
    missing_columns = [column for column in SEC_DEDUPE_COLS if column not in df]
    if missing_columns:
        logger.warning(
            'SEC reference lookup skipped; missing expected columns {}',
            missing_columns,
        )
        return empty_sec_rows()
    df = df.dropna(subset=SEC_DEDUPE_COLS)
    if df.empty:
        return empty_sec_rows()
    df = df[SEC_DEDUPE_COLS]
    df = df.sort_values('filed', kind='mergesort')
    df = df.drop_duplicates(['start', 'end'], keep='last')
    return df


def format_sec_candidate_block(matches: pd.DataFrame) -> str:
    lines = ['  SEC candidates:']
    for _, row in matches.sort_values(['end', 'start', 'filed']).iterrows():
        lines.append(
            '    '
            f'filed={row["filed"]} '
            f'period={row["start"]}..{row["end"]} '
            f'value={format_source_log_value(row["val"])}'
        )
    return '\n'.join(lines)


def select_sec_fact(
    df: pd.DataFrame,
    description: str,
    *,
    min_start: str,
    max_start: str,
    min_end: str,
    max_end: str,
    log_errors: bool = True,
) -> pd.Series | None:
    if df.empty:
        return None
    matches = df[
        df['start'].gt(min_start)
        & df['start'].lt(max_start)
        & df['end'].gt(min_end)
        & df['end'].lt(max_end)
    ]
    if matches.empty:
        if log_errors:
            logger.warning(
                'SEC fact lookup returned no matches: {} start=({}, {}) end=({}, {})',
                description,
                min_start,
                max_start,
                min_end,
                max_end,
            )
        return None
    if len(matches) > 1:
        if log_errors:
            logger.warning(
                'SEC fact lookup returned multiple matches: {} start=({}, {}) end=({}, {}) matches={}\n{}',
                description,
                min_start,
                max_start,
                min_end,
                max_end,
                len(matches),
                format_sec_candidate_block(matches),
            )
        return None
    return matches.iloc[-1]


def select_sec_quarter_fact(
    df: pd.DataFrame,
    description: str,
    end: pd.Timestamp,
    *,
    log_errors: bool = True,
) -> pd.Series | None:
    min_start = date_str(end - pd.DateOffset(months=4))
    max_start = date_str(end - pd.DateOffset(months=2))
    min_end, max_end = quarter_end_bounds(end)
    return select_sec_fact(
        df,
        description=description,
        min_start=min_start,
        max_start=max_start,
        min_end=min_end,
        max_end=max_end,
        log_errors=log_errors,
    )


def derive_sec_q4_value(
    field: str,
    df: pd.DataFrame,
    description_prefix: str,
    end: pd.Timestamp,
    *,
    log_errors: bool = True,
) -> int | float | None:
    annual = select_sec_fact(
        df,
        description=f'{description_prefix} annual {date_str(end)}',
        min_start=date_str(end - pd.DateOffset(months=13)),
        max_start=date_str(end - pd.DateOffset(months=11)),
        min_end=date_str(end - pd.DateOffset(months=1)),
        max_end=date_str(end + pd.DateOffset(months=1)),
        log_errors=log_errors,
    )
    if annual is None:
        return None

    annual_start = pd.Timestamp(annual['start'])
    q1_to_q3 = select_sec_fact(
        df,
        description=f'{description_prefix} Q1-Q3 from {annual["start"]}',
        min_start=date_str(annual_start - pd.DateOffset(months=1)),
        max_start=date_str(annual_start + pd.DateOffset(months=1)),
        min_end=date_str(annual_start + pd.DateOffset(months=8)),
        max_end=date_str(annual_start + pd.DateOffset(months=10)),
        log_errors=log_errors,
    )
    if q1_to_q3 is None:
        return None
    if field == 'weightedAverageShsOutDil':
        return round(annual['val'] * 4 - q1_to_q3['val'] * 3)
    return annual['val'] - q1_to_q3['val']


def lookup_sec_field_value(
    field: str,
    metadata: dict[str, str],
    sec_frames: list[pd.DataFrame],
    *,
    log_errors: bool = False,
) -> int | float | None:
    end = pd.Timestamp(metadata['date'])
    period = metadata.get('period')
    for df in sec_frames:
        datum = select_sec_quarter_fact(
            df,
            description=f'CIK{metadata["cik"]} {field} {date_str(end)}',
            end=end,
            log_errors=log_errors and period != 'Q4',
        )
        if datum is not None:
            return datum['val']
        if period != 'Q4':
            continue

        value = derive_sec_q4_value(
            field,
            df,
            f'CIK{metadata["cik"]} {field}',
            end,
            log_errors=log_errors,
        )
        if value is not None:
            return value
    return None


async def fetch_sec_company_concept_raw(cik: str, concept: str) -> dict:
    text = await shared.cached_get(
        SEC_COMPANY_CONCEPT_URL.format(cik, concept),
        headers={'User-Agent': SEC_USER_AGENT},
    )
    return json.loads(text)


async def fetch_sec_concept_rows(cik: str, concept: str, unit: str) -> pd.DataFrame:
    try:
        data = await fetch_sec_company_concept_raw(cik, concept)
    except HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.warning(
                'SEC concept {} not found for CIK {}; skipping', concept, cik
            )
            return empty_sec_rows()
        logger.warning(
            'SEC concept {} fetch failed for CIK {} status={}; skipping',
            concept,
            cik,
            exc.response.status_code,
        )
        return empty_sec_rows()
    except Exception as exc:
        logger.warning(
            'SEC concept {} fetch failed for CIK {}: {}; skipping',
            concept,
            cik,
            exc,
        )
        return empty_sec_rows()
    unit_rows = data.get('units', {}).get(unit, [])
    if len(unit_rows) == 0:
        logger.warning(
            'SEC concept {} response had empty units["{}"] data for CIK {}; skipping',
            concept,
            unit,
            cik,
        )
        return empty_sec_rows()
    return dedupe_sec_rows(unit_rows)


async def fetch_sec_field_rows(cik: str, field: str) -> list[pd.DataFrame]:
    frames = []
    for concept, unit in SEC_FIELD_CONCEPTS[field]:
        try:
            frame = await fetch_sec_concept_rows(cik, concept, unit)
        except Exception as exc:
            logger.warning(
                'SEC concept {} processing failed for CIK {}: {}; skipping',
                concept,
                cik,
                exc,
            )
            continue
        if not frame.empty:
            frames.append(frame)
    return frames


async def fetch_fmp_income_statements(
    market: Literal['j', 't', 'u'],
    symbol: str,
    limit: int,
) -> dict[str, dict]:
    params = {
        'apikey': FMP_KEY,
        'limit': limit,
        'period': 'quarter',
    }
    if market == 'u':
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
        eps_field = 'epsDiluted'
    else:
        url = (
            f'https://financialmodelingprep.com/api/v3/income-statement/'
            f'{add_suffix(market, symbol)}'
        )
        eps_field = 'epsdiluted'
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'weightedAverageShsOutDil',
        EPS_XPS_FIELD: eps_field,
    }
    text = await shared.cached_get(url, params)
    return normalize_income_statement_rows(
        json.loads(text),
        field_map=field_map,
        date_field='date',
        metadata_fields={
            'cik': 'cik',
            'quarter': 'period',
        },
    )


async def fetch_massive_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        'tickers': symbol,
        'timeframe': 'quarterly',
        'limit': limit,
        'sort': 'period_end.desc',
        'apiKey': MASSIVE_API_KEY,
    }
    text = await shared.cached_get(
        'https://api.massive.com/stocks/financials/v1/income-statements', params
    )
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'diluted_shares_outstanding',
        EPS_XPS_FIELD: 'diluted_earnings_per_share',
    }
    return normalize_income_statement_rows(
        json.loads(text)['results'] or [],
        field_map=field_map,
        date_field='period_end',
        metadata_fields={
            'cik': 'cik',
            'quarter': 'fiscal_quarter',
        },
    )


async def fetch_finnhub_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        'symbol': symbol,
        'statement': 'ic',
        'freq': 'quarterly',
        'preliminary': 'false',
        'token': FINNHUB_API_KEY,
    }
    text = await shared.cached_get('https://finnhub.io/api/v1/stock/financials', params)

    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'dilutedAverageSharesOutstanding',
        EPS_XPS_FIELD: 'dilutedEPS',
    }
    financials = sorted(
        json.loads(text)['financials'] or [],
        key=lambda row: row['period'],
        reverse=True,
    )
    financials = financials[:limit]
    return normalize_income_statement_rows(
        financials,
        field_map=field_map,
        date_field='period',
        value_transform=normalize_finnhub_field_value,
    )


def tiingo_fundamentals_start_date(
    limit: int, quarter_buffer: int = TIINGO_QUARTER_BUFFER
) -> str:
    start = pd.Timestamp.now(
        tz=shared.MARKET_TO_TIMEZONE['u']
    ).normalize() - pd.DateOffset(months=3 * (limit + quarter_buffer))
    return date_str(start.tz_localize(None))


def normalize_tiingo_income_statement_rows(statements: list[dict]) -> dict[str, dict]:
    """Normalize Tiingo's nested quarterly income-statement response."""
    rows = {}
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'shareswaDil',
        EPS_XPS_FIELD: 'epsDil',
    }
    for statement in statements:
        if normalize_massive_fiscal_quarter(statement.get('quarter')) is None:
            continue
        statement_data = statement.get('statementData', {})
        if not isinstance(statement_data, dict):
            continue
        income_statement = statement_data.get('incomeStatement', [])
        if not isinstance(income_statement, list) or 'date' not in statement:
            continue
        values = {
            entry.get('dataCode'): entry.get('value')
            for entry in income_statement
            if isinstance(entry, dict) and entry.get('dataCode') is not None
        }
        date = date_str(pd.Timestamp(statement['date']))
        rows[date] = normalize_source_row(values, field_map)
    return rows


async def fetch_tiingo_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    params = {
        # Use Tiingo's latest restated values, whose dates are fiscal period ends.
        'asReported': 'false',
        'startDate': tiingo_fundamentals_start_date(limit),
    }
    text = await shared.cached_get(
        TIINGO_FUNDAMENTALS_STATEMENTS_URL.format(symbol),
        params,
        headers={'Authorization': f'Token {TIINGO_API_KEY}'},
    )
    return normalize_tiingo_income_statement_rows(json.loads(text))


def diluted_eps_from_reported_income(
    income_statement_row: dict, diluted_shares: int | float | None
) -> float | None:
    if not diluted_shares:
        return None
    net_income_applicable = to_source_number(
        income_statement_row.get('netIncomeApplicableToCommonShares')
    )
    if net_income_applicable is not None:
        earnings = net_income_applicable
    else:
        net_income = to_source_number(income_statement_row.get('netIncome'))
        preferred_adjustments = to_source_number(
            income_statement_row.get('preferredStockAndOtherAdjustments')
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
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
        'filter': (
            'Financials::Income_Statement::quarterly,'
            'Financials::Balance_Sheet::quarterly'
        ),
    }
    text = await shared.cached_get(
        EODHD_FUNDAMENTALS_URL.format(f'{symbol}.US'), params
    )
    return json.loads(text)


async def fetch_eodhd_income_statements(symbol: str, limit: int) -> dict[str, dict]:
    data = await fetch_eodhd_fundamentals(symbol)
    income_statement = data.get('Financials::Income_Statement::quarterly', {})
    balance_sheet = data.get('Financials::Balance_Sheet::quarterly', {})

    rows = {}
    for date, income_row in sorted(income_statement.items(), reverse=True):
        if not isinstance(income_row, dict):
            continue

        balance_sheet_row = select_eodhd_balance_sheet_row(date, balance_sheet) or {}
        diluted_shares = to_source_number(
            balance_sheet_row.get('commonStockSharesOutstanding')
        )
        normalized_row = {
            'revenue': sanitize_source_field_value(
                'revenue', to_source_number(income_row.get('totalRevenue'))
            ),
            'weightedAverageShsOutDil': sanitize_source_field_value(
                'weightedAverageShsOutDil', diluted_shares
            ),
        }
        normalized_row[EPS_XPS_FIELD] = sanitize_source_field_value(
            EPS_XPS_FIELD,
            diluted_eps_from_reported_income(income_row, diluted_shares),
        )
        rows[date] = normalized_row
        if len(rows) >= limit:
            break
    return rows


def field_has_consensus(field: str, lhs: int | float, rhs: int | float) -> bool:
    log_diff = source_log_diff(lhs, rhs)
    if field == 'epsDiluted':
        return (
            source_abs_diff(lhs, rhs) < EPS_ABS_TOLERANCE
            or log_diff < SOURCE_DIFF_THRESHOLD
        )
    return log_diff < SOURCE_DIFF_THRESHOLD


def usable_source_values(
    source_values: dict[str, int | float | None],
    source_names: tuple[str, ...] | None = None,
) -> list[tuple[str, int | float]]:
    if source_names is None:
        source_names = us_income_statement_source_order(include_sec=True)
    values: list[tuple[str, int | float]] = []
    for source_name in source_names:
        value = source_values.get(source_name)
        if not has_source_field_value(value):
            continue
        values.append((source_name, value))
    return values


def build_consensus_groups(
    field: str,
    values: list[tuple[str, int | float]],
) -> list[tuple[tuple[str, int | float], ...]]:
    adjacent_sources = {source_name: set() for source_name, _ in values}
    for i, (lhs_name, lhs_value) in enumerate(values):
        for rhs_name, rhs_value in values[i + 1 :]:
            if not field_has_consensus(field, lhs_value, rhs_value):
                continue
            adjacent_sources[lhs_name].add(rhs_name)
            adjacent_sources[rhs_name].add(lhs_name)

    consensus_groups = []
    value_by_source = dict(values)
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
            tuple(
                (name, value_by_source[name])
                for name in us_income_statement_source_order(include_sec=True)
                if name in group_names
            )
        )
    return consensus_groups


def choose_consensus_group(
    groups: list[tuple[tuple[str, int | float], ...]],
) -> tuple[tuple[str, int | float], ...] | None:
    if not groups:
        return None

    largest_group_size = max(len(group) for group in groups)
    largest_groups = [group for group in groups if len(group) == largest_group_size]

    sec_consensus_groups = [
        group
        for group in largest_groups
        if any(source_name == 'sec' for source_name, _ in group)
    ]
    if len(largest_groups) == 1:
        return largest_groups[0]
    elif len(sec_consensus_groups) == 1:
        return sec_consensus_groups[0]
    return None


def resolve_source_consensus(
    field: str,
    source_values: dict[str, int | float | None],
) -> Consensus | None:
    values = usable_source_values(source_values)
    consensus_group = choose_consensus_group(build_consensus_groups(field, values))
    if consensus_group is None:
        return None

    return Consensus(
        value=sum(value for _, value in consensus_group) / len(consensus_group),
        sources=tuple(name for name, _ in consensus_group),
    )


def format_source_log_value(value: int | float | None) -> str:
    if value is None:
        return 'missing'
    value = float(value)
    if value.is_integer():
        return f'{int(value):,}'
    return f'{value:,.6f}'.rstrip('0').rstrip('.')


def format_source_field_block(
    field: str,
    source_values: dict[str, int | float | None],
    quarter_key: str,
) -> str:
    lines = [f'  {field}:']
    for source_name in us_income_statement_source_order(include_sec=True):
        lines.append(
            '    '
            f'{SOURCE_LABELS[source_name]:<8} '
            f'date={quarter_key:<10} '
            f'value={format_source_log_value(source_values.get(source_name))}'
        )
    return '\n'.join(lines)


def format_source_fetch_error(error: Exception) -> str:
    """Summarize provider failures without logging request URLs or API keys."""
    if isinstance(error, HTTPStatusError):
        return f'HTTP {error.response.status_code}'
    return type(error).__name__


async def fetch_us_income_statement_source(
    source: USIncomeStatementSource,
    symbol: str,
    limit: int,
) -> tuple[str, SourceIncomeStatementRows | None]:
    try:
        return source.name, await source.fetch(symbol, limit)
    except Exception as error:
        logger.warning(
            '{} income statements unavailable for {}: {}; skipping source',
            source.label,
            symbol,
            format_source_fetch_error(error),
        )
        return source.name, None


async def fetch_us_income_statement_sources(
    symbol: str, limit: int
) -> IncomeStatementFetch:
    results = await asyncio.gather(
        *(
            fetch_us_income_statement_source(source, symbol, limit)
            for source in ENABLED_US_INCOME_STATEMENT_SOURCES
        )
    )
    return IncomeStatementFetch(
        rows={name: rows if rows is not None else {} for name, rows in results},
        unavailable_sources=frozenset(name for name, rows in results if rows is None),
    )


def select_sec_cik(aligned_quarters: AlignedQuarters) -> str | None:
    for source_name in SEC_METADATA_SOURCES:
        for quarter in aligned_quarters.values():
            for _, row in quarter.source_periods.get(source_name, []):
                cik = format_sec_cik(row.get('cik'))
                if cik is not None:
                    return cik
    return None


def build_sec_quarter_metadata(
    aligned_quarters: AlignedQuarters,
    cik: str | None,
) -> dict[str, dict[str, str]]:
    if cik is None:
        return {}
    metadata = {}
    for source_name in SEC_METADATA_SOURCES:
        for quarter_key, quarter in aligned_quarters.items():
            if quarter_key in metadata:
                continue
            source_periods = quarter.source_periods.get(source_name, [])
            if not source_periods:
                continue
            source_date, row = source_periods[0]
            period = normalize_massive_fiscal_quarter(row.get('quarter'))
            if period in {'Q1', 'Q2', 'Q3', 'Q4'}:
                metadata[quarter_key] = {
                    'cik': cik,
                    'date': source_date,
                    'period': period,
                }
    return metadata


async def fetch_sec_values_for_quarters(
    symbol: str,
    cik: str,
    metadata_by_quarter: dict[str, dict[str, str]],
    fields: list[str],
) -> dict[str, dict[str, int | float]]:
    """Fetch usable SEC field values keyed by aligned quarter."""
    values_by_quarter = {}
    for field in (field for field in SEC_REFERENCE_FIELDS if field in fields):
        try:
            frames = await fetch_sec_field_rows(cik, field)
        except Exception as exc:
            logger.warning(
                'SEC reference unavailable for {} field={}: {}; continuing without SEC',
                symbol,
                field,
                exc,
            )
            continue
        if not frames:
            continue

        for quarter_key, metadata in metadata_by_quarter.items():
            value = lookup_sec_field_value(field, metadata, frames)
            value = sanitize_source_field_value(field, value)
            if value is not None:
                values_by_quarter.setdefault(quarter_key, {})[field] = value
    return values_by_quarter


async def add_sec_reference_values(
    symbol: str,
    aligned_quarters: AlignedQuarters,
    fields: list[str],
) -> None:
    """Add usable SEC facts to aligned US quarters as reference-source values."""
    cik = select_sec_cik(aligned_quarters)
    if cik is None:
        logger.warning('SEC reference unavailable for {}: no FMP/Massive CIK', symbol)
        return

    metadata_by_quarter = build_sec_quarter_metadata(aligned_quarters, cik)
    if not metadata_by_quarter:
        logger.warning(
            'SEC reference unavailable for {}: no FMP/Massive quarter metadata',
            symbol,
        )
        return

    values_by_quarter = await fetch_sec_values_for_quarters(
        symbol,
        cik,
        metadata_by_quarter,
        fields,
    )
    for quarter_key, field_values in values_by_quarter.items():
        quarter = aligned_quarters[quarter_key]
        for field, value in field_values.items():
            quarter.fields.setdefault(field, {})['sec'] = value


def drop_incomplete_latest_us_quarter(
    symbol: str,
    aligned_quarters: AlignedQuarters,
    required_fields: list[str],
) -> AlignedQuarters:
    quarter_dates = list(aligned_quarters)
    if not quarter_dates:
        return aligned_quarters

    latest_date = quarter_dates[-1]
    latest_quarter = aligned_quarters[latest_date]
    insufficient_fields = [
        field
        for field in required_fields
        if len(usable_source_values(latest_quarter.fields.get(field, {}))) < 2
    ]

    if not insufficient_fields:
        return aligned_quarters

    mismatch_blocks = [
        format_source_field_block(
            field, latest_quarter.fields.get(field, {}), latest_date
        )
        for field in required_fields
        if field in insufficient_fields
    ]
    logger.warning(
        'Dropping latest {} quarter {}: fewer than 2 usable sources\n{}',
        symbol,
        latest_date,
        '\n'.join(mismatch_blocks),
    )
    return {date: aligned_quarters[date] for date in quarter_dates[:-1]}


def resolve_us_quarter_consensus(
    symbol: str,
    anchor_date: str,
    quarter: AlignedQuarter,
    required_fields: list[str],
) -> ResolvedQuarter:
    resolved_quarter = {}
    for field in required_fields:
        source_values = quarter.fields.get(field, {})
        consensus = resolve_source_consensus(field, source_values)
        if consensus is not None:
            resolved_quarter[field] = consensus.value
            continue
        logger.warning(
            'Aborting {} quarter {}: no consensus\n{}',
            symbol,
            anchor_date,
            format_source_field_block(field, source_values, anchor_date),
        )
        raise ValueError(f'no consensus for {symbol} {anchor_date} field={field}')
    return resolved_quarter


async def resolve_us_income_statement_quarters(
    symbol: str,
    source_rows: IncomeStatementSources,
    limit: int,
    include_eps: bool,
    *,
    unavailable_sources: frozenset[str] = frozenset(),
) -> ResolvedQuarters:
    """Align US provider data, add SEC references, then resolve consensus values."""
    required_fields = required_xps_fields(include_eps)

    # 1. Group provider rows that describe the same fiscal quarter.
    aligned_quarters = build_aligned_source_quarters(source_rows)

    # 2. Add SEC shares/EPS as a reference source before validation.
    await add_sec_reference_values(
        symbol,
        aligned_quarters,
        required_fields,
    )

    # 3. Ignore an under-supported newest quarter, then retain the requested window.
    eligible_quarters = drop_incomplete_latest_us_quarter(
        symbol, aligned_quarters, required_fields
    )
    selected_quarters = select_latest_required_quarters(
        eligible_quarters, limit, symbol=symbol, quarter_label='aligned quarters'
    )

    # 4. Record diagnostics for the same quarters that will be returned.
    if include_eps:
        record_xps_diagnostics(
            symbol,
            selected_quarters,
            unavailable_sources=unavailable_sources,
        )

    # 5. Reduce each field to the mean of its winning consensus group.
    return {
        anchor_date: resolve_us_quarter_consensus(
            symbol, anchor_date, quarter, required_fields
        )
        for anchor_date, quarter in selected_quarters.items()
    }


async def fetch_resolved_income_statement_quarters(
    market: Literal['j', 't', 'u'],
    symbol: str,
    limit: int,
    include_eps: bool,
) -> ResolvedQuarters:
    if market == 't':
        return await fetch_finmind_taiwan_income_statements(
            symbol, limit, require_eps=include_eps
        )

    if market != 'u':
        fmp_rows = await fetch_fmp_income_statements(market, symbol, limit)
        return select_latest_required_quarters(fmp_rows, limit, symbol=symbol)

    fetched = await fetch_us_income_statement_sources(symbol, limit + 1)
    return await resolve_us_income_statement_quarters(
        symbol,
        fetched.rows,
        limit,
        include_eps,
        unavailable_sources=fetched.unavailable_sources,
    )


def build_xps_frame(
    resolved_quarters: ResolvedQuarters, include_eps: bool
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(resolved_quarters, orient='index')
    df.index = pd.to_datetime(df.index) + pd.Timedelta(days=1)
    frame = pd.DataFrame(
        {
            'rps': df['revenue'] / df['weightedAverageShsOutDil'],
            **({'eps': df['epsDiluted']} if include_eps else {}),
        }
    )
    return frame.rolling(4).sum().iloc[3:]


async def fetch_xps(
    market: Literal['j', 't', 'u'], symbol: str, q: int, include_eps: bool = True
) -> pd.DataFrame:
    limit = q + 3
    resolved_quarters = await fetch_resolved_income_statement_quarters(
        market, symbol, limit, include_eps
    )
    return build_xps_frame(resolved_quarters, include_eps)


async def calc_px(market: Literal['j', 't', 'u'], symbol: str, q: int) -> pd.DataFrame:
    prices, xps = await asyncio.gather(
        shared.get_prices(market, symbol, 91 * q, False),
        fetch_xps(market, symbol, q + EXTRA_Q),
    )
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(prices),
    )
    df = (
        pd.DataFrame({'price': prices}, index)
        .join(xps, how='outer')
        .ffill()
        .tail(len(index))
    )
    if (pd.isna(df['price'].iloc[0])) or pd.isna(df['rps'].iloc[0]):
        raise ValueError
    df['ps'] = df['price'] / df['rps']
    df['pe'] = df['price'] / df['eps'].where(df['eps'] > 0)
    return df[['ps', 'pe']]


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
