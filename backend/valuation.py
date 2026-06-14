import asyncio
import math
from typing import Literal

import pandas as pd
from httpx import AsyncClient
from loguru import logger

from . import shared
from .shared import (
    FINNHUB_API_KEY,
    FMP_KEY,
    MASSIVE_API_KEY,
    add_suffix,
)

EXTRA_Q = 1
SOURCE_DIFF_THRESHOLD = 0.065
SOURCE_DATE_DIFF_TOLERANCE_DAYS = 7
EPS_ABS_TOLERANCE = 0.02
FINNHUB_MILLION_SCALE = 1_000_000
SOURCE_ORDER = ('fmp', 'massive', 'finnhub')
SOURCE_LABELS = {
    'fmp': 'FMP',
    'massive': 'Massive',
    'finnhub': 'Finnhub',
}
BASE_XPS_FIELDS = ('revenue', 'weightedAverageShsOutDil')
EPS_XPS_FIELD = 'epsDiluted'
ALL_XPS_FIELDS = (*BASE_XPS_FIELDS, EPS_XPS_FIELD)


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


def normalize_fmp_income_statement_rows(
    rows: list[dict], include_eps: bool, eps_field: str
) -> dict[str, dict]:
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'weightedAverageShsOutDil',
        EPS_XPS_FIELD: eps_field,
    }
    return {
        row['date']: normalize_source_row(row, field_map, include_eps) for row in rows
    }


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
    async with AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
    return normalize_fmp_income_statement_rows(
        response.json(), include_eps=require_eps, eps_field=eps_field
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
    rows = {}
    field_map = {
        'revenue': 'revenue',
        'weightedAverageShsOutDil': 'diluted_shares_outstanding',
        EPS_XPS_FIELD: 'diluted_earnings_per_share',
    }
    for row in response.json()['results'] or []:
        period_end = row['period_end']
        rows[period_end] = normalize_source_row(row, field_map, require_eps)
    return rows


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

    rows = {}
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
    for row in financials:
        period_end = row['period']
        rows[period_end] = normalize_source_row(
            row,
            field_map,
            require_eps,
            value_transform=normalize_finnhub_field_value,
        )
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
        if not has_source_field_value(value):
            continue
        values.append((source_name, value))

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
    for source_name, rows in zip(SOURCE_ORDER, source_rows, strict=True):
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
    latest_missing_usable_fields = [
        field
        for field, source_values in latest_field_source_context.items()
        if count_usable_source_values(source_values) < 2
    ]

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


def resolve_us_income_statement_quarters(
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
    aligned_quarters = drop_incomplete_latest_us_quarter(
        symbol, aligned_quarters, required_fields
    )
    aligned_quarters = select_latest_required_quarters(aligned_quarters, limit)
    return {
        anchor_date: resolve_us_quarter_consensus(
            symbol, anchor_date, quarter, required_fields
        )
        for anchor_date, quarter in aligned_quarters.items()
    }


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
    return resolve_us_income_statement_quarters(
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
