import asyncio
import math
from typing import Literal

import pandas as pd
from httpx import AsyncClient, HTTPStatusError
from loguru import logger

from . import shared
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
BASE_SOURCE_ORDER = ('fmp', 'massive', 'finnhub')
SOURCE_ORDER = (*BASE_SOURCE_ORDER, 'sec')
SOURCE_LABELS = {
    'fmp': 'FMP',
    'massive': 'Massive',
    'finnhub': 'Finnhub',
    'sec': 'SEC',
}
BASE_XPS_FIELDS = ('revenue', 'weightedAverageShsOutDil')
EPS_XPS_FIELD = 'epsDiluted'
ALL_XPS_FIELDS = (*BASE_XPS_FIELDS, EPS_XPS_FIELD)
SEC_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
)
SEC_COMPANY_CONCEPT_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'
)
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
    return value is not None


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
    if value is None:
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


def normalize_finnhub_field_value(
    field: str, value: int | float | None
) -> int | float | None:
    if field in BASE_XPS_FIELDS:
        return finnhub_million(value)
    return value


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
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
    df = df[df['form'].isin(SEC_ALLOWED_FORMS)]
    if df.empty:
        logger.warning('SEC repair skipped; no supported forms found')
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
    missing_columns = [column for column in SEC_DEDUPE_COLS if column not in df]
    if missing_columns:
        logger.warning(
            'SEC repair skipped; missing expected columns {}',
            missing_columns,
        )
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
    df = df.dropna(subset=SEC_DEDUPE_COLS)
    if df.empty:
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
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
    if df.empty:
        return None
    matches = df
    matches = matches[matches['start'].gt(min_start)]
    matches = matches[matches['start'].lt(max_start)]
    matches = matches[matches['end'].gt(min_end)]
    matches = matches[matches['end'].lt(max_end)]
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
    log_errors: bool = True,
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
        log_errors=log_errors,
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
        log_errors=log_errors,
    )
    return annual, q1_to_q3


def derive_sec_q4_value(
    field: str, annual: pd.Series, q1_to_q3: pd.Series
) -> int | float:
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
        if period != 'Q4':
            datum = select_sec_quarter_datum(
                df,
                description=f'CIK{metadata["cik"]} {field} {date_str(end)}',
                end=end,
                log_errors=log_errors,
            )
            if datum is not None:
                return datum['val']
            continue

        datum = select_sec_quarter_datum(
            df,
            description=f'CIK{metadata["cik"]} {field} Q4 exact {date_str(end)}',
            end=end,
            log_errors=False,
        )
        if datum is not None:
            return datum['val']

        annual, q1_to_q3 = select_sec_q4_components(
            df,
            annual_description=(f'CIK{metadata["cik"]} {field} annual {date_str(end)}'),
            q1_to_q3_description_prefix=(f'CIK{metadata["cik"]} {field} Q1-Q3 from '),
            end=end,
            log_errors=log_errors,
        )
        if annual is not None and q1_to_q3 is not None:
            return derive_sec_q4_value(field, annual, q1_to_q3)
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
            return pd.DataFrame(columns=SEC_DEDUPE_COLS)
        logger.warning(
            'SEC concept {} fetch failed for CIK {} status={}; skipping',
            concept,
            cik,
            exc.response.status_code,
        )
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
    except Exception as exc:
        logger.warning(
            'SEC concept {} fetch failed for CIK {}: {}; skipping',
            concept,
            cik,
            exc,
        )
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
    unit_rows = data.get('units', {}).get(unit, [])
    if len(unit_rows) == 0:
        logger.warning(
            'SEC concept {} response had empty units["{}"] data for CIK {}; skipping',
            concept,
            unit,
            cik,
        )
        return pd.DataFrame(columns=SEC_DEDUPE_COLS)
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
        if not has_source_field_value(value):
            continue
        values.append((source_name, value))

    passing_pairs = []
    adjacent_sources = {source_name: set() for source_name, _ in values}
    for i, (lhs_name, lhs_value) in enumerate(values):
        for rhs_name, rhs_value in values[i + 1 :]:
            if not field_has_consensus(field, lhs_value, rhs_value):
                continue
            passing_pairs.append(((lhs_name, lhs_value), (rhs_name, rhs_value)))
            adjacent_sources[lhs_name].add(rhs_name)
            adjacent_sources[rhs_name].add(lhs_name)

    if not passing_pairs:
        return None, None

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

    largest_group_size = max(len(group) for group in consensus_groups)
    consensus_groups = [
        group for group in consensus_groups if len(group) == largest_group_size
    ]

    if len(consensus_groups) == 1:
        consensus_group = consensus_groups[0]
        return (
            sum(value for _, value in consensus_group) / len(consensus_group),
            tuple(name for name, _ in consensus_group),
        )

    return None, None


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


async def fetch_us_income_statement_sources(
    symbol: str, limit: int, include_eps: bool
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    source_rows = await asyncio.gather(
        fetch_fmp_income_statements('u', symbol, limit, require_eps=include_eps),
        fetch_massive_income_statements(symbol, limit, require_eps=include_eps),
        fetch_finnhub_income_statements(symbol, limit, require_eps=include_eps),
        return_exceptions=True,
    )
    resolved_rows = []
    for source_name, rows in zip(BASE_SOURCE_ORDER, source_rows, strict=True):
        if isinstance(rows, Exception):
            logger.warning(
                '{} income statements unavailable for {}: {}; skipping source',
                SOURCE_LABELS[source_name],
                symbol,
                rows,
            )
            rows = {}
        resolved_rows.append(rows)
    return tuple(resolved_rows)


def select_sec_cik(
    fmp_rows: dict[str, dict], massive_rows: dict[str, dict]
) -> str | None:
    if fmp_rows:
        return format_sec_cik(next(iter(fmp_rows.values()))['cik'])
    for row in massive_rows.values():
        cik = format_sec_cik(row['cik'])
        if cik is not None:
            return cik
    return None


def build_sec_quarter_metadata_from_source(
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    source_rows: dict[str, dict],
    cik: str,
) -> dict[str, dict[str, str]]:
    metadata = {}
    for source_date, row in sorted(source_rows.items()):
        quarter_key = select_closest_aligned_quarter_key(
            source_date, list(aligned_quarters)
        )
        if quarter_key is None:
            continue
        period = normalize_massive_fiscal_quarter(row.get('quarter'))
        if period not in {'Q1', 'Q2', 'Q3', 'Q4'}:
            continue
        metadata[quarter_key] = {
            'cik': cik,
            'date': source_date,
            'period': period,
        }
    return metadata


def build_sec_quarter_metadata(
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    fmp_rows: dict[str, dict],
    massive_rows: dict[str, dict],
    cik: str | None,
) -> dict[str, dict[str, str]]:
    if cik is None:
        return {}
    metadata = build_sec_quarter_metadata_from_source(aligned_quarters, fmp_rows, cik)
    fallback_metadata = build_sec_quarter_metadata_from_source(
        aligned_quarters, massive_rows, cik
    )
    for quarter_key, quarter_metadata in fallback_metadata.items():
        metadata.setdefault(quarter_key, quarter_metadata)
    return metadata


class SecIncomeStatementSource:
    def __init__(
        self,
        symbol: str,
        fmp_rows: dict[str, dict],
        massive_rows: dict[str, dict],
        aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    ) -> None:
        self.symbol = symbol
        self.cik = select_sec_cik(fmp_rows, massive_rows)
        self.metadata = build_sec_quarter_metadata(
            aligned_quarters, fmp_rows, massive_rows, self.cik
        )
        self.loaded_fields: set[str] = set()
        self.field_frames: dict[str, list[pd.DataFrame]] = {}

    def has_loaded(self, field: str) -> bool:
        return field in self.loaded_fields

    async def load_field(self, field: str) -> None:
        if self.has_loaded(field):
            return
        self.loaded_fields.add(field)
        if self.cik is None:
            logger.warning(
                'SEC repair unavailable for {}: no FMP/Massive CIK',
                self.symbol,
            )
            self.field_frames[field] = []
            return
        if not self.metadata:
            logger.warning(
                'SEC repair unavailable for {}: no FMP/Massive quarter metadata',
                self.symbol,
            )
            self.field_frames[field] = []
            return
        try:
            self.field_frames[field] = await fetch_sec_field_rows(self.cik, field)
        except Exception as exc:
            logger.warning(
                'SEC repair unavailable for {} field={}: {}; continuing without SEC',
                self.symbol,
                field,
                exc,
            )
            self.field_frames[field] = []

    async def merge_field(
        self,
        aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
        field: str,
    ) -> None:
        await self.load_field(field)
        frames = self.field_frames.get(field, [])
        if not frames:
            return
        for quarter_key, quarter in aligned_quarters.items():
            metadata = self.metadata.get(quarter_key)
            if metadata is None:
                continue
            value = lookup_sec_field_value(field, metadata, frames)
            value = sanitize_source_field_value(field, value)
            if value is None:
                continue
            quarter.setdefault(field, {})['sec'] = value

    async def merge_fields(
        self,
        aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
        fields: list[str],
    ) -> None:
        for field in dict.fromkeys(fields):
            await self.merge_field(aligned_quarters, field)


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


def find_us_consensus_failure(
    aligned_quarters: dict[str, dict[str, dict[str, int | float | None]]],
    required_fields: list[str],
) -> tuple[str, str] | None:
    for anchor_date, quarter in aligned_quarters.items():
        for field in required_fields:
            _, consensus_sources = select_consensus_source_value(
                field, quarter.get(field, {})
            )
            if consensus_sources is None:
                return anchor_date, field
    return None


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


async def resolve_us_income_statement_quarters(
    symbol: str,
    fmp_rows: dict[str, dict],
    massive_rows: dict[str, dict],
    finnhub_rows: dict[str, dict],
    limit: int,
    include_eps: bool,
) -> dict[str, dict[str, int | float | None]]:
    required_fields = required_xps_fields(include_eps)
    aligned_quarters = build_aligned_source_quarters(
        fmp_rows, massive_rows, finnhub_rows
    )
    sec_source = SecIncomeStatementSource(
        symbol, fmp_rows, massive_rows, aligned_quarters
    )
    await sec_source.merge_fields(
        aligned_quarters,
        latest_insufficient_us_fields(aligned_quarters, required_fields),
    )
    aligned_quarters = drop_incomplete_latest_us_quarter(
        symbol, aligned_quarters, required_fields
    )
    aligned_quarters = select_latest_required_quarters(aligned_quarters, limit)
    while True:
        consensus_failure = find_us_consensus_failure(aligned_quarters, required_fields)
        if consensus_failure is None:
            return resolve_all_us_quarter_consensus(
                symbol, aligned_quarters, required_fields
            )

        _, field = consensus_failure
        if sec_source.has_loaded(field):
            return resolve_all_us_quarter_consensus(
                symbol, aligned_quarters, required_fields
            )

        await sec_source.merge_field(aligned_quarters, field)


async def fetch_resolved_income_statement_quarters(
    market: Literal['j', 't', 'u'],
    symbol: str,
    limit: int,
    include_eps: bool,
) -> dict[str, dict[str, int | float | None]]:
    if market != 'u':
        fmp_rows = await fetch_fmp_income_statements(
            market, symbol, limit, require_eps=include_eps
        )
        return select_latest_required_quarters(fmp_rows, limit)

    source_rows = await fetch_us_income_statement_sources(
        symbol, limit + 1, include_eps
    )
    return await resolve_us_income_statement_quarters(
        symbol, *source_rows, limit, include_eps
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
