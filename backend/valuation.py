import asyncio
import math
from collections import Counter
from itertools import combinations
from typing import Literal

import pandas as pd
from httpx import AsyncClient, HTTPStatusError
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
    cached,
)

EXTRA_Q = 1
SOURCE_DIFF_THRESHOLD = 0.065
SOURCE_MATCH_THRESHOLD = 0.02
SOURCE_DATE_DIFF_TOLERANCE_DAYS = 7
EPS_ABS_TOLERANCE = 0.02
FINNHUB_MILLION_SCALE = 1_000_000
BASE_SOURCE_ORDER = (
    'fmp',
    *(("massive",) if shared.ENABLE_MASSIVE_FUNDAMENTALS else ()),
    *(("eodhd",) if shared.ENABLE_EODHD_FUNDAMENTALS else ()),
    'finnhub',
    *(("tiingo",) if shared.ENABLE_TIINGO_FUNDAMENTALS else ()),
)
SOURCE_ORDER = (*BASE_SOURCE_ORDER, 'sec')
SOURCE_LABELS = {
    'fmp': 'FMP',
    'massive': 'Massive',
    'eodhd': 'EODHD',
    'finnhub': 'Finnhub',
    'tiingo': 'Tiingo',
    'sec': 'SEC',
}

# Coverage is intentionally process-local. Restarting the backend starts a new
# screener run with a clean counter.
_xps_coverage_points = Counter()
_xps_coverage_observations = 0
_xps_coverage_seen: set[tuple[str, str]] = set()
BASE_XPS_FIELDS = ('revenue', 'weightedAverageShsOutDil')
EPS_XPS_FIELD = 'epsDiluted'
ALL_XPS_FIELDS = (*BASE_XPS_FIELDS, EPS_XPS_FIELD)
# SEC is a repair/reference source, not a peer provider for source comparison.
XPS_SOURCE_PAIRS = tuple(combinations(BASE_SOURCE_ORDER, 2))
_xps_coverage_complete_keys = {source: set() for source in SOURCE_ORDER}
_xps_coverage_pair_stats = {
    pair: {
        'both_complete': 0,
        'all_fields_exact': 0,
        'all_fields_within_2_percent': 0,
        'all_fields_within_consensus': 0,
        'fields': {
            field: {
                'both_present': 0,
                'exact_matches': 0,
                'within_2_percent': 0,
                'within_consensus': 0,
            }
            for field in ALL_XPS_FIELDS
        },
    }
    for pair in XPS_SOURCE_PAIRS
}
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
SEC_FIELD_CONCEPTS = {
    'revenue': (
        ('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD'),
        ('Revenues', 'USD'),
    ),
    'weightedAverageShsOutDil': (
        ('WeightedAverageNumberOfDilutedSharesOutstanding', 'shares'),
    ),
    EPS_XPS_FIELD: (('EarningsPerShareDiluted', 'USD/shares'),),
}


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


def source_values_within_match_threshold(lhs: int | float, rhs: int | float) -> bool:
    lhs = float(lhs)
    rhs = float(rhs)
    if not all(map(math.isfinite, (lhs, rhs))):
        return False
    scale = max(abs(lhs), abs(rhs))
    if scale == 0:
        return True
    return abs(lhs - rhs) / scale <= SOURCE_MATCH_THRESHOLD


def record_xps_coverage(
    symbol: str,
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    required_fields: list[str],
) -> None:
    """Count complete source quarters once for the current backend run."""
    global _xps_coverage_observations

    for quarter, data in aligned_quarters.items():
        key = (symbol, quarter)
        if key in _xps_coverage_seen:
            continue

        _xps_coverage_seen.add(key)
        _xps_coverage_observations += 1
        source_complete = {}
        for source in SOURCE_ORDER:
            complete = all(
                has_source_field_value(data.get(field, {}).get(source))
                for field in required_fields
            )
            source_complete[source] = complete
            _xps_coverage_points[source] += int(complete)
            if complete:
                _xps_coverage_complete_keys[source].add(key)

        for pair in XPS_SOURCE_PAIRS:
            lhs_source, rhs_source = pair
            pair_stats = _xps_coverage_pair_stats[pair]
            if source_complete[lhs_source] and source_complete[rhs_source]:
                pair_stats['both_complete'] += 1

            all_fields_exact = True
            all_fields_within_2_percent = True
            all_fields_within_consensus = True
            for field in required_fields:
                lhs = data.get(field, {}).get(lhs_source)
                rhs = data.get(field, {}).get(rhs_source)
                if not (has_source_field_value(lhs) and has_source_field_value(rhs)):
                    all_fields_exact = False
                    all_fields_within_2_percent = False
                    all_fields_within_consensus = False
                    continue

                field_stats = pair_stats['fields'][field]
                field_stats['both_present'] += 1
                field_stats['exact_matches'] += int(lhs == rhs)
                within_2_percent = source_values_within_match_threshold(lhs, rhs)
                field_stats['within_2_percent'] += int(within_2_percent)
                within_consensus = field_has_consensus(field, lhs, rhs)
                field_stats['within_consensus'] += int(within_consensus)
                all_fields_exact &= lhs == rhs
                all_fields_within_2_percent &= within_2_percent
                all_fields_within_consensus &= within_consensus

            if source_complete[lhs_source] and source_complete[rhs_source]:
                pair_stats['all_fields_exact'] += int(all_fields_exact)
                pair_stats['all_fields_within_2_percent'] += int(
                    all_fields_within_2_percent
                )
                pair_stats['all_fields_within_consensus'] += int(
                    all_fields_within_consensus
                )


def get_xps_coverage() -> dict:
    """Return aggregate source coverage for the current backend process."""
    source_coverage = {}
    for source in SOURCE_ORDER:
        complete_keys = _xps_coverage_complete_keys[source]
        source_coverage[source] = {
            'points': _xps_coverage_points[source],
            'ratio': (
                _xps_coverage_points[source] / _xps_coverage_observations
                if _xps_coverage_observations
                else 0
            ),
        }

    comparisons = {}
    for lhs_source, rhs_source in XPS_SOURCE_PAIRS:
        lhs_keys = _xps_coverage_complete_keys[lhs_source]
        rhs_keys = _xps_coverage_complete_keys[rhs_source]
        intersection = lhs_keys & rhs_keys
        union = lhs_keys | rhs_keys
        pair_stats = _xps_coverage_pair_stats[(lhs_source, rhs_source)]
        values = {}
        for field, field_stats in pair_stats['fields'].items():
            both_present = field_stats['both_present']
            values[field] = {
                **field_stats,
                'exact_ratio': (
                    field_stats['exact_matches'] / both_present if both_present else 0
                ),
                'match_ratio': (
                    field_stats['within_2_percent'] / both_present
                    if both_present
                    else 0
                ),
                'consensus_ratio': (
                    field_stats['within_consensus'] / both_present
                    if both_present
                    else 0
                ),
            }
        comparisons[f'{lhs_source}:{rhs_source}'] = {
            'coverage': {
                'left_points': len(lhs_keys),
                'right_points': len(rhs_keys),
                'overlap_points': len(intersection),
                'left_only_points': len(lhs_keys - rhs_keys),
                'right_only_points': len(rhs_keys - lhs_keys),
                'jaccard_ratio': len(intersection) / len(union) if union else 0,
            },
            'complete_rows': {
                'both_complete': pair_stats['both_complete'],
                'all_fields_exact': pair_stats['all_fields_exact'],
                'exact_ratio': (
                    pair_stats['all_fields_exact'] / pair_stats['both_complete']
                    if pair_stats['both_complete']
                    else 0
                ),
                'all_fields_within_2_percent': pair_stats[
                    'all_fields_within_2_percent'
                ],
                'match_ratio': (
                    pair_stats['all_fields_within_2_percent']
                    / pair_stats['both_complete']
                    if pair_stats['both_complete']
                    else 0
                ),
                'all_fields_within_consensus': pair_stats[
                    'all_fields_within_consensus'
                ],
                'consensus_ratio': (
                    pair_stats['all_fields_within_consensus']
                    / pair_stats['both_complete']
                    if pair_stats['both_complete']
                    else 0
                ),
            },
            'fields': values,
        }

    return {
        'observations': _xps_coverage_observations,
        'sources': source_coverage,
        'comparisons': comparisons,
    }


def select_closest_aligned_quarter_key(
    source_date: str, quarter_keys: list[str]
) -> str | None:
    source_ts = pd.Timestamp(source_date)
    matches = []
    for quarter_key in quarter_keys:
        delta_days = abs((pd.Timestamp(quarter_key) - source_ts).days)
        if delta_days >= SOURCE_DATE_DIFF_TOLERANCE_DAYS:
            continue
        matches.append((delta_days, quarter_key))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[0][1]


def align_source_rows(
    aligned_quarters: dict[str, dict[str, dict[str, float]]],
    source_name: str,
    source_rows: dict[str, dict],
) -> None:
    for source_date, row in sorted(source_rows.items()):
        quarter_key = select_closest_aligned_quarter_key(
            source_date, list(aligned_quarters)
        )
        if quarter_key is None:
            quarter_key = source_date
            aligned_quarters[quarter_key] = {}
        quarter = aligned_quarters[quarter_key]
        for field in ALL_XPS_FIELDS:
            if field not in row:
                continue
            quarter.setdefault(field, {})[source_name] = row[field]


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
    include_eps: bool,
    value_transform=lambda field, value: value,
) -> dict[str, int | float | None]:
    return {
        field: sanitize_source_field_value(
            field,
            value_transform(field, row.get(source_field)),
        )
        for field, source_field in field_map.items()
        if include_eps or field != EPS_XPS_FIELD
    }


def normalize_income_statement_rows(
    rows: list[dict],
    *,
    field_map: dict[str, str],
    date_field: str,
    include_eps: bool,
    metadata_fields: dict[str, str] | None = None,
    value_transform=lambda field, value: value,
) -> dict[str, dict]:
    normalized_rows = {}
    for row in rows:
        normalized_row = normalize_source_row(
            row,
            field_map,
            include_eps,
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

    aligned_date = select_closest_aligned_quarter_key(income_date, list(balance_sheet))
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
    *,
    include_eps: bool,
) -> dict[str, dict]:
    financial_statement = pivot_finmind_taiwan_rows(financial_statement_rows)
    balance_sheet = pivot_finmind_taiwan_rows(balance_sheet_rows)
    if financial_statement.empty or balance_sheet.empty:
        return {}

    required_financial_statement_fields = ['Revenue']
    if include_eps:
        required_financial_statement_fields.append('EPS')
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
            'FinMind Taiwan balance sheet data missing OrdinaryShare; skipping rows'
        )
        balance_sheet = pd.DataFrame(index=financial_statement.index)
        balance_sheet['OrdinaryShare'] = None

    df = financial_statement[required_financial_statement_fields].join(
        balance_sheet[['OrdinaryShare']], how='outer'
    )
    df = df.dropna(subset=required_financial_statement_fields, how='all')

    normalized_rows = {}
    for date, row in df.iterrows():
        normalized_row = {
            'revenue': sanitize_source_field_value('revenue', row['Revenue']),
            'weightedAverageShsOutDil': sanitize_source_field_value(
                'weightedAverageShsOutDil',
                row['OrdinaryShare'] / FINMIND_TAIWAN_SHARE_PAR_VALUE,
            ),
        }
        if include_eps:
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
        baseline_date = select_closest_aligned_quarter_key(
            preferred_date, list(baseline_rows)
        )
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


@cached(43200)
async def fetch_finmind_taiwan_rows(
    dataset: str, symbol: str, start_date: str
) -> list[dict]:
    async with AsyncClient(timeout=60) as client:
        response = await client.get(
            FINMIND_API_URL,
            params={
                'dataset': dataset,
                'data_id': symbol,
                'start_date': start_date,
                'token': FINMIND_KEY,
            },
        )
    response.raise_for_status()
    return response.json().get('data', [])


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
            require_eps=require_eps,
        ),
    )
    finmind_rows = normalize_finmind_taiwan_income_statement_rows(
        financial_statement_rows,
        balance_sheet_rows,
        include_eps=require_eps,
    )
    merge_preferred_xps_rows(rows, finmind_rows)
    rows = drop_incomplete_latest_quarter(rows, required_fields)
    return select_latest_clean_required_quarters(
        rows, limit, required_fields, symbol=symbol
    )


def build_aligned_source_quarters(
    source_rows: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    aligned_quarters = {}
    for source_name in BASE_SOURCE_ORDER:
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
        logger.warning('SEC repair skipped; no supported forms found')
        return empty_sec_rows()
    missing_columns = [column for column in SEC_DEDUPE_COLS if column not in df]
    if missing_columns:
        logger.warning(
            'SEC repair skipped; missing expected columns {}',
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


@cached(43200)
async def fetch_sec_company_concept_raw(cik: str, concept: str) -> dict:
    async with AsyncClient(headers={'User-Agent': SEC_USER_AGENT}) as client:
        response = await client.get(SEC_COMPANY_CONCEPT_URL.format(cik, concept))
        response.raise_for_status()
        return response.json()


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
    require_eps: bool = True,
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
    async with AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
    return normalize_income_statement_rows(
        response.json(),
        field_map=field_map,
        date_field='date',
        include_eps=require_eps,
        metadata_fields={
            'cik': 'cik',
            'quarter': 'period',
        },
    )


async def fetch_massive_income_statements(
    symbol: str, limit: int, require_eps: bool = True
) -> dict[str, dict]:
    params = {
        'tickers': symbol,
        'timeframe': 'quarterly',
        'limit': limit,
        'sort': 'period_end.desc',
        'apiKey': MASSIVE_API_KEY,
    }
    async with AsyncClient() as client:
        response = await client.get(
            'https://api.massive.com/stocks/financials/v1/income-statements',
            params=params,
        )
        response.raise_for_status()
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'diluted_shares_outstanding',
        EPS_XPS_FIELD: 'diluted_earnings_per_share',
    }
    return normalize_income_statement_rows(
        response.json()['results'] or [],
        field_map=field_map,
        date_field='period_end',
        include_eps=require_eps,
        metadata_fields={
            'cik': 'cik',
            'quarter': 'fiscal_quarter',
        },
    )


async def fetch_finnhub_income_statements(
    symbol: str, limit: int | None = None, require_eps: bool = True
) -> dict[str, dict]:
    params = {
        'symbol': symbol,
        'statement': 'ic',
        'freq': 'quarterly',
        'token': FINNHUB_API_KEY,
    }
    async with AsyncClient(timeout=30) as client:
        response = await client.get(
            'https://finnhub.io/api/v1/stock/financials',
            params=params,
        )
        response.raise_for_status()

    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'dilutedAverageSharesOutstanding',
        EPS_XPS_FIELD: 'dilutedEPS',
    }
    financials = sorted(
        response.json()['financials'] or [],
        key=lambda row: row['period'],
        reverse=True,
    )
    if limit is not None:
        financials = financials[:limit]
    return normalize_income_statement_rows(
        financials,
        field_map=field_map,
        date_field='period',
        include_eps=require_eps,
        value_transform=normalize_finnhub_field_value,
    )


def tiingo_fundamentals_start_date(
    limit: int, quarter_buffer: int = TIINGO_QUARTER_BUFFER
) -> str:
    start = pd.Timestamp.now(
        tz=shared.MARKET_TO_TIMEZONE['u']
    ).normalize() - pd.DateOffset(months=3 * (limit + quarter_buffer))
    return date_str(start.tz_localize(None))


def normalize_tiingo_income_statement_rows(
    statements: list[dict], *, include_eps: bool
) -> dict[str, dict]:
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
        rows[date] = normalize_source_row(values, field_map, include_eps)
    return rows


async def fetch_tiingo_income_statements(
    symbol: str, limit: int, require_eps: bool = True
) -> dict[str, dict]:
    params = {
        # Use Tiingo's latest restated values, whose dates are fiscal period ends.
        'asReported': 'false',
        'startDate': tiingo_fundamentals_start_date(limit),
    }
    async with AsyncClient(
        timeout=30, headers={'Authorization': f'Token {TIINGO_API_KEY}'}
    ) as client:
        response = await client.get(
            TIINGO_FUNDAMENTALS_STATEMENTS_URL.format(symbol), params=params
        )
        response.raise_for_status()
    return normalize_tiingo_income_statement_rows(
        response.json(), include_eps=require_eps
    )


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


@cached(43200)
async def fetch_eodhd_fundamentals(symbol: str) -> dict:
    params = {
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
        'filter': (
            'Financials::Income_Statement::quarterly,'
            'Financials::Balance_Sheet::quarterly'
        ),
    }
    async with AsyncClient(timeout=30) as client:
        response = await client.get(
            EODHD_FUNDAMENTALS_URL.format(f'{symbol}.US'),
            params=params,
        )
        response.raise_for_status()
    return response.json()


async def fetch_eodhd_income_statements(
    symbol: str, limit: int | None = None, require_eps: bool = True
) -> dict[str, dict]:
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
        if require_eps:
            normalized_row[EPS_XPS_FIELD] = sanitize_source_field_value(
                EPS_XPS_FIELD,
                diluted_eps_from_reported_income(income_row, diluted_shares),
            )
        rows[date] = normalized_row
        if limit is not None and len(rows) >= limit:
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
) -> list[tuple[str, int | float]]:
    values: list[tuple[str, int | float]] = []
    for source_name in SOURCE_ORDER:
        value = source_values.get(source_name)
        if not has_source_field_value(value):
            continue
        values.append((source_name, value))
    return values


def build_consensus_groups(
    field: str, values: list[tuple[str, int | float]]
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
                for name in SOURCE_ORDER
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


def average_consensus_group(group: tuple[tuple[str, int | float], ...]) -> float:
    return sum(value for _, value in group) / len(group)


def select_consensus_source_value(
    field: str, source_values: dict[str, int | float | None]
) -> tuple[int | float | None, tuple[str, ...] | None]:
    values = usable_source_values(source_values)
    consensus_group = choose_consensus_group(build_consensus_groups(field, values))
    if consensus_group is None:
        return None, None

    return (
        average_consensus_group(consensus_group),
        tuple(name for name, _ in consensus_group),
    )


def count_usable_source_values(source_values: dict[str, int | float | None]) -> int:
    return sum(
        has_source_field_value(source_values.get(source_name))
        for source_name in SOURCE_ORDER
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
    for source_name in SOURCE_ORDER:
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


async def fetch_us_income_statement_sources(
    symbol: str, limit: int, include_eps: bool
) -> dict[str, dict[str, dict]]:
    source_fetches = {
        'fmp': fetch_fmp_income_statements('u', symbol, limit, require_eps=include_eps),
        'finnhub': fetch_finnhub_income_statements(
            symbol, limit, require_eps=include_eps
        ),
    }
    if shared.ENABLE_MASSIVE_FUNDAMENTALS:
        source_fetches['massive'] = fetch_massive_income_statements(
            symbol, limit, require_eps=include_eps
        )
    if shared.ENABLE_EODHD_FUNDAMENTALS:
        source_fetches['eodhd'] = fetch_eodhd_income_statements(
            symbol, limit, require_eps=include_eps
        )
    if shared.ENABLE_TIINGO_FUNDAMENTALS:
        source_fetches['tiingo'] = fetch_tiingo_income_statements(
            symbol, limit, require_eps=include_eps
        )
    source_rows = await asyncio.gather(
        *source_fetches.values(),
        return_exceptions=True,
    )
    resolved_rows = []
    for source_name, rows in zip(source_fetches, source_rows, strict=True):
        if isinstance(rows, Exception):
            logger.warning(
                '{} income statements unavailable for {}: {}; skipping source',
                SOURCE_LABELS[source_name],
                symbol,
                format_source_fetch_error(rows),
            )
            rows = {}
        resolved_rows.append((source_name, rows))
    return dict(resolved_rows)


def select_sec_cik(
    fmp_rows: dict[str, dict], massive_rows: dict[str, dict]
) -> str | None:
    for source_rows in (fmp_rows, massive_rows):
        for row in source_rows.values():
            cik = format_sec_cik(row.get('cik'))
            if cik is not None:
                return cik
    return None


def build_sec_quarter_metadata(
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    fmp_rows: dict[str, dict],
    massive_rows: dict[str, dict],
    cik: str | None,
) -> dict[str, dict[str, str]]:
    if cik is None:
        return {}
    metadata = {}
    quarter_keys = list(aligned_quarters)
    for source_rows in (fmp_rows, massive_rows):
        for source_date, row in sorted(source_rows.items()):
            quarter_key = select_closest_aligned_quarter_key(source_date, quarter_keys)
            if quarter_key is None or quarter_key in metadata:
                continue
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
    for field in dict.fromkeys(fields):
        if field == 'revenue':
            continue
        try:
            frames = await fetch_sec_field_rows(cik, field)
        except Exception as exc:
            logger.warning(
                'SEC repair unavailable for {} field={}: {}; continuing without SEC',
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


async def merge_sec_fields(
    symbol: str,
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    fmp_rows: dict[str, dict],
    massive_rows: dict[str, dict],
    fields: list[str],
) -> None:
    cik = select_sec_cik(fmp_rows, massive_rows)
    if cik is None:
        logger.warning('SEC repair unavailable for {}: no FMP/Massive CIK', symbol)
        return

    metadata_by_quarter = build_sec_quarter_metadata(
        aligned_quarters, fmp_rows, massive_rows, cik
    )
    if not metadata_by_quarter:
        logger.warning(
            'SEC repair unavailable for {}: no FMP/Massive quarter metadata',
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
            quarter.setdefault(field, {})['sec'] = value


def latest_insufficient_us_fields(
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    required_fields: list[str],
) -> list[str]:
    quarter_dates = list(aligned_quarters)
    if not quarter_dates:
        return []

    latest_quarter = aligned_quarters[quarter_dates[-1]]
    return [
        field
        for field in required_fields
        if count_usable_source_values(latest_quarter.get(field, {})) < 2
    ]


def drop_incomplete_latest_us_quarter(
    symbol: str,
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    required_fields: list[str],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    quarter_dates = list(aligned_quarters)
    if not quarter_dates:
        return aligned_quarters

    latest_date = quarter_dates[-1]
    latest_quarter = aligned_quarters[latest_date]
    latest_field_source_context = {
        field: latest_quarter.get(field, {}) for field in required_fields
    }
    latest_missing_usable_fields = latest_insufficient_us_fields(
        aligned_quarters, required_fields
    )

    if not latest_missing_usable_fields:
        return aligned_quarters

    mismatch_blocks = [
        format_source_field_block(
            field, latest_field_source_context[field], latest_date
        )
        for field in required_fields
        if field in latest_missing_usable_fields
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
    quarter: dict[str, dict[str, int | float | None]],
    required_fields: list[str],
) -> dict[str, int | float | None]:
    resolved_quarter = {}
    for field in required_fields:
        source_values = quarter.get(field, {})
        accepted_value, consensus_sources = select_consensus_source_value(
            field, source_values
        )
        if consensus_sources is not None:
            resolved_quarter[field] = accepted_value
            continue
        logger.warning(
            'Aborting {} quarter {}: no consensus\n{}',
            symbol,
            anchor_date,
            format_source_field_block(field, source_values, anchor_date),
        )
        raise ValueError(f'no consensus for {symbol} {anchor_date} field={field}')
    return resolved_quarter


def resolve_all_us_quarter_consensus(
    symbol: str,
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    required_fields: list[str],
) -> dict[str, dict[str, int | float | None]]:
    return {
        anchor_date: resolve_us_quarter_consensus(
            symbol, anchor_date, quarter, required_fields
        )
        for anchor_date, quarter in aligned_quarters.items()
    }


async def prepare_us_income_statement_quarters(
    symbol: str,
    source_rows: dict[str, dict[str, dict]],
    required_fields: list[str],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    aligned_quarters = build_aligned_source_quarters(source_rows)
    await merge_sec_fields(
        symbol,
        aligned_quarters,
        source_rows['fmp'],
        source_rows.get('massive', {}),
        required_fields,
    )
    return drop_incomplete_latest_us_quarter(symbol, aligned_quarters, required_fields)


async def resolve_us_income_statement_quarters(
    symbol: str,
    source_rows: dict[str, dict[str, dict]],
    limit: int,
    include_eps: bool,
) -> dict[str, dict[str, int | float | None]]:
    required_fields = required_xps_fields(include_eps)
    aligned_quarters = await prepare_us_income_statement_quarters(
        symbol, source_rows, required_fields
    )
    aligned_quarters = select_latest_required_quarters(
        aligned_quarters, limit, symbol=symbol, quarter_label='aligned quarters'
    )
    if include_eps:
        record_xps_coverage(symbol, aligned_quarters, required_fields)
    return resolve_all_us_quarter_consensus(symbol, aligned_quarters, required_fields)


async def fetch_resolved_income_statement_quarters(
    market: Literal['j', 't', 'u'],
    symbol: str,
    limit: int,
    include_eps: bool,
) -> dict[str, dict[str, int | float | None]]:
    if market == 't':
        return await fetch_finmind_taiwan_income_statements(
            symbol, limit, require_eps=include_eps
        )

    if market != 'u':
        fmp_rows = await fetch_fmp_income_statements(
            market, symbol, limit, require_eps=include_eps
        )
        return select_latest_required_quarters(fmp_rows, limit, symbol=symbol)

    source_rows = await fetch_us_income_statement_sources(
        symbol, limit + 1, include_eps
    )
    return await resolve_us_income_statement_quarters(
        symbol,
        source_rows,
        limit,
        include_eps,
    )


def build_xps_frame(
    resolved_quarters: dict[str, dict[str, int | float | None]], include_eps: bool
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(resolved_quarters, orient='index')
    out = {
        'rps': (df['revenue'] / df['weightedAverageShsOutDil'])
        .rolling(4)
        .sum()
        .to_numpy()
    }
    if include_eps:
        out['eps'] = df['epsDiluted'].rolling(4).sum().to_numpy()
    return pd.DataFrame(
        out,
        pd.to_datetime(df.index) + pd.Timedelta(days=1),
    ).iloc[3:]


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
