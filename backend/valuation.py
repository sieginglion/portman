import asyncio
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from httpx import AsyncClient, HTTPStatusError
from loguru import logger
from pydantic import BaseModel

from . import shared
from .llm import llm
from .shared import (
    FINNHUB_API_KEY,
    FMP_KEY,
    MASSIVE_API_KEY,
    add_suffix,
    cached,
)

EXTRA_Q = 1
SOURCE_DIFF_THRESHOLD = 0.065
SOURCE_DATE_DIFF_TOLERANCE_DAYS = 7
EPS_ABS_TOLERANCE = 0.02
FINNHUB_MILLION_SCALE = 1_000_000
PATCH_DIR = Path('patch')
SOURCE_ORDER = ('fmp', 'massive', 'finnhub')
SOURCE_LABELS = {
    'fmp': 'FMP',
    'massive': 'Massive',
    'finnhub': 'Finnhub',
}
PERIOD_ORDER = ('Q1', 'Q2', 'Q3', 'Q4')
SEC_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
)
SEC_DEDUPE_COLS = ['filed', 'val', 'start', 'end']
SEC_ALLOWED_FORMS = {'10-Q', '10-K', '10-Q/A', '10-K/A'}
SEC_DILUTED_SHARES_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/'
    'CIK{}/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json'
)
SEC_EPS_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/'
    'CIK{}/us-gaap/EarningsPerShareDiluted.json'
)


class IncomeStatementSourceFields(BaseModel):
    revenue: int | float | None = None
    weightedAverageShsOutDil: int | float | None = None
    epsDiluted: float | None = None


class IncomeStatementVerificationRequest(BaseModel):
    symbol: str
    date: str
    fiscalYear: str | None = None
    period: str | None = None
    cik: str | None = None
    issue: str
    fmp_statement: IncomeStatementSourceFields | None = None
    massive_statement: IncomeStatementSourceFields | None = None
    finnhub_statement: IncomeStatementSourceFields | None = None


class VerifiedIncomeStatementFields(BaseModel):
    revenue: int | float | None = None
    weightedAverageShsOutDil: int | float | None = None
    epsDiluted: float | None = None


def review_fields(
    row: dict | None, eps_col: str, fields: list[str] | set[str] | None = None
) -> IncomeStatementSourceFields | None:
    if row is None:
        return None
    fields = set(fields or ('revenue', 'weightedAverageShsOutDil', 'epsDiluted'))
    return IncomeStatementSourceFields(
        revenue=row.get('revenue') if 'revenue' in fields else None,
        weightedAverageShsOutDil=(
            row.get('weightedAverageShsOutDil')
            if 'weightedAverageShsOutDil' in fields
            else None
        ),
        epsDiluted=row.get(eps_col) if 'epsDiluted' in fields else None,
    )


def date_str(date: pd.Timestamp) -> str:
    return date.strftime('%Y-%m-%d')


def quarter_end_bounds(end: pd.Timestamp, months: int = 1) -> tuple[str, str]:
    return (
        date_str(end - pd.DateOffset(months=months)),
        date_str(end + pd.DateOffset(months=months)),
    )


def dedupe_sec_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[df['form'].isin(SEC_ALLOWED_FORMS)]
    if df.empty:
        logger.warning('SEC repair skipped; no supported forms found')
        return pd.DataFrame()
    df = df[SEC_DEDUPE_COLS]
    df = df.sort_values('filed', kind='mergesort')
    df = df.drop_duplicates(['start', 'end'], keep='last')
    return df


def select_sec_datum(
    df: pd.DataFrame,
    description: str,
    *,
    min_start: str,
    max_start: str,
    min_end: str,
    max_end: str,
    log_errors: bool = True,
) -> pd.Series | None:
    matches = df
    matches = matches[matches['start'].gt(min_start)]
    matches = matches[matches['start'].lt(max_start)]
    matches = matches[matches['end'].gt(min_end)]
    matches = matches[matches['end'].lt(max_end)]
    if matches.empty:
        if log_errors:
            logger.error(
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
            logger.error(
                'SEC fact lookup returned multiple matches: {} start=({}, {}) end=({}, {}) matches={}',
                description,
                min_start,
                max_start,
                min_end,
                max_end,
                len(matches),
            )
        return None
    return matches.iloc[-1]


def select_sec_quarter_datum(
    df: pd.DataFrame,
    description: str,
    end: pd.Timestamp,
    *,
    log_errors: bool = True,
) -> pd.Series | None:
    min_start = date_str(end - pd.DateOffset(months=4))
    max_start = date_str(end - pd.DateOffset(months=2))
    min_end, max_end = quarter_end_bounds(end)
    return select_sec_datum(
        df,
        description=description,
        min_start=min_start,
        max_start=max_start,
        min_end=min_end,
        max_end=max_end,
        log_errors=log_errors,
    )


def select_sec_q4_components(
    df: pd.DataFrame,
    *,
    annual_description: str,
    q1_to_q3_description_prefix: str,
    end: pd.Timestamp,
) -> tuple[pd.Series | None, pd.Series | None]:
    min_start = date_str(end - pd.DateOffset(months=13))
    max_start = date_str(end - pd.DateOffset(months=11))
    annual = select_sec_datum(
        df,
        description=annual_description,
        min_start=min_start,
        max_start=max_start,
        min_end=date_str(end - pd.DateOffset(months=1)),
        max_end=date_str(end + pd.DateOffset(months=1)),
    )
    if annual is None:
        return None, None

    annual_start = pd.Timestamp(annual['start'])
    min_end = date_str(annual_start + pd.DateOffset(months=8))
    max_end = date_str(annual_start + pd.DateOffset(months=10))
    min_start = date_str(annual_start - pd.DateOffset(months=1))
    max_start = date_str(annual_start + pd.DateOffset(months=1))
    q1_to_q3 = select_sec_datum(
        df,
        description=f'{q1_to_q3_description_prefix}{annual["start"]}',
        min_start=min_start,
        max_start=max_start,
        min_end=min_end,
        max_end=max_end,
    )
    return annual, q1_to_q3


def lookup_sec_diluted_shares(row: dict, shares: pd.DataFrame) -> int | None:
    end = pd.Timestamp(row['date'])
    if row['period'] != 'Q4':
        datum = select_sec_quarter_datum(
            shares,
            description=f'CIK{row["cik"]} {date_str(end)}',
            end=end,
        )
        if datum is None:
            return None
        return datum['val']

    datum = select_sec_quarter_datum(
        shares,
        description=f'CIK{row["cik"]} Q4 exact {date_str(end)}',
        end=end,
        log_errors=False,
    )
    if datum is not None:
        return datum['val']

    annual, q1_to_q3 = select_sec_q4_components(
        shares,
        annual_description=f'CIK{row["cik"]} annual {date_str(end)}',
        q1_to_q3_description_prefix=f'CIK{row["cik"]} Q1-Q3 from ',
        end=end,
    )
    if annual is None or q1_to_q3 is None:
        return None
    return round(annual['val'] * 4 - q1_to_q3['val'] * 3)


def find_sec_diluted_shares(row: dict, shares: pd.DataFrame) -> int:
    if (value := lookup_sec_diluted_shares(row, shares)) is None:
        return row['weightedAverageShsOutDil']
    return value


def lookup_sec_diluted_eps(row: dict, eps: pd.DataFrame, eps_col: str) -> float | None:
    end = pd.Timestamp(row['date'])
    if row['period'] != 'Q4':
        datum = select_sec_quarter_datum(
            eps,
            description=f'CIK{row["cik"]} EPS {date_str(end)}',
            end=end,
        )
        if datum is None:
            return None
        return datum['val']

    datum = select_sec_quarter_datum(
        eps,
        description=f'CIK{row["cik"]} EPS Q4 exact {date_str(end)}',
        end=end,
        log_errors=False,
    )
    if datum is not None:
        return datum['val']

    annual, q1_to_q3 = select_sec_q4_components(
        eps,
        annual_description=f'CIK{row["cik"]} EPS annual {date_str(end)}',
        q1_to_q3_description_prefix=f'CIK{row["cik"]} EPS Q1-Q3 from ',
        end=end,
    )
    if annual is None or q1_to_q3 is None:
        return None
    return annual['val'] - q1_to_q3['val']


def find_sec_diluted_eps(row: dict, eps: pd.DataFrame, eps_col: str) -> float:
    if (value := lookup_sec_diluted_eps(row, eps, eps_col)) is None:
        return row[eps_col]
    return value


@cached(43200)
async def _fetch_sec_diluted_shares_raw(cik: int | str) -> dict:
    async with AsyncClient(headers={'User-Agent': SEC_USER_AGENT}) as client:
        response = await client.get(
            SEC_DILUTED_SHARES_URL.format(cik),
        )
        response.raise_for_status()
        return response.json()


async def fetch_sec_diluted_shares(cik: int | str) -> pd.DataFrame:
    try:
        data = await _fetch_sec_diluted_shares_raw(cik)
    except HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.error(
                'SEC diluted-share concept not found (404); keeping FMP value for CIK {}',
                cik,
            )
            return pd.DataFrame()
        logger.error(
            'SEC diluted-share fetch failed for CIK {} status={}; aborting',
            cik,
            exc.response.status_code,
        )
        raise
    unit_rows = data['units']['shares']
    if len(unit_rows) == 0:
        logger.error(
            'SEC diluted-share response had empty units["shares"] data; skipping CIK {}',
            cik,
        )
        return pd.DataFrame()
    return dedupe_sec_rows(unit_rows)


@cached(43200)
async def _fetch_sec_eps_raw(cik: int | str) -> dict:
    async with AsyncClient(headers={'User-Agent': SEC_USER_AGENT}) as client:
        response = await client.get(
            SEC_EPS_URL.format(cik),
        )
        response.raise_for_status()
        return response.json()


async def fetch_sec_eps(cik: int | str) -> pd.DataFrame:
    try:
        data = await _fetch_sec_eps_raw(cik)
    except HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.error(
                'SEC diluted EPS concept not found (404); keeping FMP value for CIK {}',
                cik,
            )
            return pd.DataFrame()
        logger.error(
            'SEC diluted EPS fetch failed for CIK {} status={}; aborting',
            cik,
            exc.response.status_code,
        )
        raise
    unit_rows = data['units']['USD/shares']
    if len(unit_rows) == 0:
        logger.error(
            'SEC diluted EPS response had empty units["USD/shares"] data; skipping CIK {}',
            cik,
        )
        return pd.DataFrame()
    return dedupe_sec_rows(unit_rows)


# TODO
def share_count_log_diff(row: dict, eps_col: str) -> float:
    eps = row[eps_col]
    shs = row['weightedAverageShsOutDil']
    net_income = row['netIncome']
    if not all(map(math.isfinite, (net_income, eps, shs))):
        return math.inf
    if eps == 0 or shs == 0:
        return math.inf
    ratio = net_income / eps / shs
    if ratio <= 0:
        return math.inf
    return abs(math.log(ratio))


def maybe_float(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def first_numeric(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = maybe_float(row.get(key))
        if value is not None:
            return value
    return None


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


def is_valid_source_field_value(field: str, value: int | float | None) -> bool:
    if value is None:
        return False
    if field in {'revenue', 'weightedAverageShsOutDil'}:
        return value > 0
    return True


def next_period_info(
    period: str, fiscal_year: str | int | None
) -> tuple[str, str] | None:
    if period not in PERIOD_ORDER or fiscal_year is None:
        return None
    fiscal_year_int = int(fiscal_year)
    index = PERIOD_ORDER.index(period)
    next_period = PERIOD_ORDER[(index + 1) % len(PERIOD_ORDER)]
    if period == 'Q4':
        fiscal_year_int += 1
    return next_period, str(fiscal_year_int)


def prev_period_info(
    period: str, fiscal_year: str | int | None
) -> tuple[str, str] | None:
    if period not in PERIOD_ORDER or fiscal_year is None:
        return None
    fiscal_year_int = int(fiscal_year)
    index = PERIOD_ORDER.index(period)
    prev_period = PERIOD_ORDER[(index - 1) % len(PERIOD_ORDER)]
    if period == 'Q1':
        fiscal_year_int -= 1
    return prev_period, str(fiscal_year_int)


def select_closest_aligned_quarter_key(
    source_date: str, quarter_keys: list[str]
) -> str | None:
    source_ts = pd.Timestamp(source_date)
    matches = []
    for quarter_key in quarter_keys:
        delta_days = abs((pd.Timestamp(quarter_key) - source_ts).days)
        if delta_days > SOURCE_DATE_DIFF_TOLERANCE_DAYS:
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
        for field in ('revenue', 'weightedAverageShsOutDil', 'epsDiluted'):
            if field not in row:
                continue
            quarter.setdefault(field, {})[source_name] = row[field]


def normalize_fmp_income_statement_rows(
    rows: list[dict], include_eps: bool = True
) -> dict[str, dict]:
    normalized_rows = {}
    for row in rows:
        if (date := row.get('date')) is None:
            continue
        normalized = {}
        if (revenue := row.get('revenue')) is not None:
            normalized['revenue'] = revenue
        if (shares := row.get('weightedAverageShsOutDil')) is not None:
            normalized['weightedAverageShsOutDil'] = shares
        if include_eps:
            eps = row.get('epsDiluted', row.get('epsdiluted'))
            if eps is not None:
                normalized['epsDiluted'] = eps
        normalized_rows.setdefault(date, normalized)
    return normalized_rows


def build_aligned_source_quarters(
    fmp_rows: dict[str, dict],
    massive_rows: dict[str, dict],
    finnhub_rows: dict[str, dict],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    aligned_quarters = {}
    align_source_rows(aligned_quarters, 'fmp', fmp_rows)
    align_source_rows(aligned_quarters, 'massive', massive_rows)
    align_source_rows(aligned_quarters, 'finnhub', finnhub_rows)
    return dict(sorted(aligned_quarters.items()))


def select_latest_required_quarters(
    quarters: dict[str, dict], limit: int
) -> dict[str, dict]:
    selected = dict(sorted(quarters.items())[-limit:])
    if len(selected) != limit:
        raise ValueError
    return selected


@cached(240)
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
    rows = {}
    for row in response.json().get('results', []):
        normalized = {}
        if (revenue := row.get('revenue')) is not None:
            normalized['revenue'] = revenue
        if (eps := row.get('diluted_earnings_per_share')) is not None:
            normalized['epsDiluted'] = eps
        if (shares := row.get('diluted_shares_outstanding')) is not None:
            normalized['weightedAverageShsOutDil'] = shares
        rows.setdefault(
            row['period_end'],
            normalized,
        )
    return rows


@cached(240)
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
        try:
            response = await client.get(
                'https://finnhub.io/api/v1/stock/financials',
                params=params,
            )
            response.raise_for_status()
        except HTTPStatusError as exc:
            if exc.response.status_code in {401, 402, 403, 404}:
                logger.warning(
                    'Finnhub income statements unavailable for {} status={}; skipping source',
                    symbol,
                    exc.response.status_code,
                )
                return {}
            raise

    rows = {}
    financials = sorted(
        response.json().get('financials', []),
        key=lambda row: row.get('period') or row.get('date') or '',
        reverse=True,
    )
    if limit is not None:
        financials = financials[:limit]
    for row in financials:
        period_end = row.get('period') or row.get('date')
        if not period_end:
            continue
        normalized = {}
        if (revenue := row.get('revenue')) is not None:
            normalized['revenue'] = revenue * FINNHUB_MILLION_SCALE
        if (eps := row.get('dilutedEPS')) is not None:
            normalized['epsDiluted'] = eps
        if (shares := row.get('dilutedAverageSharesOutstanding')) is not None:
            normalized['weightedAverageShsOutDil'] = shares * FINNHUB_MILLION_SCALE
        rows.setdefault(period_end, normalized)
    return rows


def field_has_consensus(field: str, lhs: int | float, rhs: int | float) -> bool:
    log_diff = source_log_diff(lhs, rhs)
    if field == 'epsDiluted':
        return (
            source_abs_diff(lhs, rhs) < EPS_ABS_TOLERANCE
            or log_diff < SOURCE_DIFF_THRESHOLD
        )
    return log_diff < SOURCE_DIFF_THRESHOLD


def select_consensus_source_value(
    field: str, source_values: dict[str, int | float | None]
) -> tuple[int | float | None, tuple[str, ...] | None]:
    values = []
    for source_name in SOURCE_ORDER:
        value = source_values.get(source_name)
        if not is_valid_source_field_value(field, value):
            continue
        values.append((source_name, value))

    if len(values) < 2:
        return None, None

    passing_pairs = []
    for i, (lhs_name, lhs_value) in enumerate(values):
        for rhs_name, rhs_value in values[i + 1 :]:
            if not field_has_consensus(field, lhs_value, rhs_value):
                continue
            passing_pairs.append(((lhs_name, lhs_value), (rhs_name, rhs_value)))

    if len(passing_pairs) == 3:
        return (
            sum(value for _, value in values) / len(values),
            tuple(name for name, _ in values),
        )

    if len(passing_pairs) == 2:
        source_counts = {}
        source_values = {}
        for lhs, rhs in passing_pairs:
            for source_name, source_value in (lhs, rhs):
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                source_values[source_name] = source_value
        shared_sources = [
            source_name for source_name, count in source_counts.items() if count == 2
        ]
        if len(shared_sources) == 1:
            shared_source = shared_sources[0]
            return source_values[shared_source], (shared_source,)

    if len(passing_pairs) == 1:
        lhs, rhs = passing_pairs[0]
        return ((lhs[1] + rhs[1]) / 2, (lhs[0], rhs[0]))

    return None, None


def count_usable_source_values(
    field: str, source_values: dict[str, int | float | None]
) -> int:
    count = 0
    for source_name in SOURCE_ORDER:
        if is_valid_source_field_value(field, source_values.get(source_name)):
            count += 1
    return count


def format_source_field_details(
    field: str,
    source_values: dict[str, int | float | None],
    quarter_key: str,
) -> str:
    details = []
    for source_name in SOURCE_ORDER:
        value = source_values.get(source_name)
        details.append(f'{SOURCE_LABELS[source_name]}[{quarter_key}]={value}')
    return f'{field}: ' + ', '.join(details)


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


@llm
async def verify_income_statement_with_agent(
    statement: IncomeStatementVerificationRequest,
) -> VerifiedIncomeStatementFields:
    """
    Verify all fields for the exact symbol, fiscal year, and period. FMP,
    Massive, and Finnhub may all be wrong. May use authoritative financial
    news sources. Return corrected values only for fields that are incorrect.
    """
    ...


# TODO
async def verify_income_statement_diluted_shares(
    row: dict,
) -> dict[str, int | float]:
    data = await fetch_sec_diluted_shares(row['cik'])
    if data.empty:
        return {'weightedAverageShsOutDil': row['weightedAverageShsOutDil']}
    return {'weightedAverageShsOutDil': find_sec_diluted_shares(row, data)}


async def verify_income_statement_eps(
    row: dict, eps_col: str
) -> dict[str, int | float]:
    data = await fetch_sec_eps(row['cik'])
    if data.empty:
        return {eps_col: row[eps_col]}
    return {eps_col: find_sec_diluted_eps(row, data, eps_col)}


async def maybe_patch_row_from_sec(
    row: dict, eps_col: str, mismatch_fields: list[str]
) -> tuple[dict, set[str]]:
    if (
        row.get('reportedCurrency') != 'USD'
        or row.get('cik') is None
        or row.get('period') is None
    ):
        return row, set()

    patched = dict(row)
    resolved_fields = set()
    sec_tasks = []
    if 'weightedAverageShsOutDil' in mismatch_fields:
        sec_tasks.append(
            ('weightedAverageShsOutDil', fetch_sec_diluted_shares(row['cik']))
        )
    if 'epsDiluted' in mismatch_fields:
        sec_tasks.append(('epsDiluted', fetch_sec_eps(row['cik'])))
    if not sec_tasks:
        return patched, resolved_fields

    sec_results = await asyncio.gather(*(task for _, task in sec_tasks))
    for field, data in zip((field for field, _ in sec_tasks), sec_results):
        if data.empty:
            continue
        if field == 'weightedAverageShsOutDil':
            value = lookup_sec_diluted_shares(patched, data)
            if value is None:
                continue
            patched['weightedAverageShsOutDil'] = value
            resolved_fields.add(field)
        else:
            value = lookup_sec_diluted_eps(patched, data, eps_col)
            if value is None:
                continue
            patched[eps_col] = value
            resolved_fields.add(field)

    return patched, resolved_fields


# @cached(240)
async def fetch_xps(
    market: Literal['j', 't', 'u'], symbol: str, q: int, include_eps: bool = True
) -> pd.DataFrame:
    limit = q + 3
    source_limit = q + 4
    eps_col = None
    params = {
        'apikey': FMP_KEY,
        'limit': source_limit if market == 'u' else limit,
        'period': 'quarter',
    }
    if market == 'u':
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
    else:
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ add_suffix(market, symbol) }'
    if include_eps:
        eps_col = 'epsDiluted'
    if market == 'u':
        async with AsyncClient() as client:
            fmp_response, massive_rows, finnhub_rows = await asyncio.gather(
                client.get(url, params=params),
                fetch_massive_income_statements(
                    symbol, source_limit, require_eps=include_eps
                ),
                fetch_finnhub_income_statements(
                    symbol, source_limit, require_eps=include_eps
                ),
                return_exceptions=True,
            )
        if isinstance(massive_rows, Exception):
            logger.warning(
                'Massive income statements unavailable for {}: {}; skipping source',
                symbol,
                massive_rows,
            )
            massive_rows = {}
        if isinstance(finnhub_rows, Exception):
            logger.warning(
                'Finnhub income statements unavailable for {}: {}; skipping source',
                symbol,
                finnhub_rows,
            )
            finnhub_rows = {}
        if isinstance(fmp_response, Exception):
            logger.warning(
                'FMP income statements unavailable for {}: {}; skipping source',
                symbol,
                fmp_response,
            )
            data = []
        else:
            try:
                fmp_response.raise_for_status()
            except HTTPStatusError as exc:
                logger.warning(
                    'FMP income statements unavailable for {} status={}; skipping source',
                    symbol,
                    exc.response.status_code,
                )
                data = []
            else:
                data = fmp_response.json()
    else:
        async with AsyncClient() as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    # path = PATCH_DIR / f'{symbol}.json'
    # if path.exists():
    #     with path.open() as f:
    #         patch = json.load(f)
    #     data = sorted(
    #         {r['date']: r for r in data + patch}.values(),
    #         key=lambda r: r['date'],
    #         reverse=True,
    #     )[:limit]
    resolved_quarters = {}
    if market == 'u':
        fmp_rows = normalize_fmp_income_statement_rows(data, include_eps)
        aligned_quarters = build_aligned_source_quarters(
            fmp_rows, massive_rows, finnhub_rows
        )
        quarter_dates = list(aligned_quarters)
        mismatch_fields = ['revenue', 'weightedAverageShsOutDil']
        if include_eps:
            mismatch_fields.append('epsDiluted')
        if quarter_dates:
            latest_date = quarter_dates[-1]
            latest_quarter = aligned_quarters[latest_date]
            latest_field_source_context = {}
            latest_missing_usable_fields = []
            for field in mismatch_fields:
                source_values = latest_quarter.get(field, {})
                latest_field_source_context[field] = source_values
                if count_usable_source_values(field, source_values) < 2:
                    latest_missing_usable_fields.append(field)
            if latest_missing_usable_fields:
                mismatch_blocks = [
                    format_source_field_block(
                        field, latest_field_source_context[field], latest_date
                    )
                    for field in mismatch_fields
                    if field in latest_missing_usable_fields
                ]
                logger.warning(
                    'Dropping latest {} quarter {}: fewer than 2 usable sources\n{}',
                    symbol,
                    latest_date,
                    '\n'.join(mismatch_blocks),
                )
                quarter_dates = quarter_dates[:-1]

        aligned_quarters = select_latest_required_quarters(
            {date: aligned_quarters[date] for date in quarter_dates}, limit
        )
        for anchor_date, quarter in aligned_quarters.items():
            field_source_context = {}

            resolved_quarter = {}
            for field in mismatch_fields:
                source_values = field_source_context.get(field)
                if source_values is None:
                    source_values = quarter.get(field, {})
                    field_source_context[field] = source_values
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
                    format_source_field_block(
                        field, field_source_context[field], anchor_date
                    ),
                )
                raise ValueError(
                    f'no consensus for {symbol} {anchor_date} field={field}'
                )

            resolved_quarters[anchor_date] = resolved_quarter
    else:
        resolved_quarters = normalize_fmp_income_statement_rows(data, include_eps)
        resolved_quarters = select_latest_required_quarters(resolved_quarters, limit)

    df = pd.DataFrame.from_dict(resolved_quarters, orient='index')
    out = {
        'rps': (df['revenue'] / df['weightedAverageShsOutDil'])
        .rolling(4)
        .sum()
        .to_numpy()
    }
    if include_eps:
        out['eps'] = df[eps_col].rolling(4).sum().to_numpy()
    return pd.DataFrame(
        out,
        pd.to_datetime(df.index) + pd.Timedelta(days=1),
    ).iloc[3:]


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


async def calc_pegs(market: Literal['j', 't', 'u'], symbol: str, q: int) -> pd.Series:
    prices, xps = await asyncio.gather(
        shared.get_prices(market, symbol, 91 * q, False),
        fetch_xps(market, symbol, q + EXTRA_Q + 4),
    )
    xps['eps_growth'] = xps['eps'].pct_change(4)
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(prices),
    )
    df = (
        pd.DataFrame({'price': prices}, index)
        .join(xps, how='outer')
        .ffill()
        .tail(91 * q)
    )
    # if (
    #     (len(df) != 91 * q)
    #     or (pd.isna(df['price'].iloc[0]))
    #     or pd.isna(df['eps'].iloc[0])
    #     or pd.isna(df['eps_growth'].iloc[0])
    # ):
    #     raise ValueError

    # df = df[(df['eps'] > 0) & (df['eps_growth'] > 0)]
    return df['price'] / df['eps'] / (df['eps_growth'] * 100)


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
