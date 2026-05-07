import asyncio
import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from httpx import AsyncClient, HTTPStatusError
from loguru import logger

from . import shared
from .shared import FMP_KEY, add_suffix, cached

EXTRA_Q = 1
SHARE_COUNT_DIFF_THRESHOLD = 0.04
PATCH_DIR = Path('patch')
SEC_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
)
SEC_DEDUPE_COLS = ['filed', 'val', 'start', 'end']
SEC_FOREIGN_FORMS = {'20-F', '20-F/A'}
SEC_DILUTED_SHARES_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/'
    'CIK{}/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json'
)
SEC_EPS_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/'
    'CIK{}/us-gaap/EarningsPerShareDiluted.json'
)


def date_str(date: pd.Timestamp) -> str:
    return date.strftime('%Y-%m-%d')


def quarter_end_bounds(end: pd.Timestamp, days: int = 7) -> tuple[str, str]:
    return (
        date_str(end - pd.DateOffset(days=days)),
        date_str(end + pd.DateOffset(days=days)),
    )


def dedupe_sec_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df['form'].isin(SEC_FOREIGN_FORMS).any():
        logger.warning(
            'SEC repair skipped for foreign filer; forms={}',
            sorted(set(df['form'].dropna())),
        )
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
) -> pd.Series | None:
    matches = df
    matches = matches[matches['start'].gt(min_start)]
    matches = matches[matches['start'].lt(max_start)]
    matches = matches[matches['end'].gt(min_end)]
    matches = matches[matches['end'].lt(max_end)]
    if matches.empty:
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
    df: pd.DataFrame, description: str, end: pd.Timestamp
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
        min_end=date_str(end - pd.DateOffset(days=7)),
        max_end=date_str(end + pd.DateOffset(days=7)),
    )
    if annual is None:
        return None, None

    annual_start = pd.Timestamp(annual['start'])
    min_end = date_str(annual_start + pd.DateOffset(months=8))
    max_end = date_str(annual_start + pd.DateOffset(months=10))
    min_start = date_str(annual_start - pd.DateOffset(days=7))
    max_start = date_str(annual_start + pd.DateOffset(days=7))
    q1_to_q3 = select_sec_datum(
        df,
        description=f'{q1_to_q3_description_prefix}{annual["start"]}',
        min_start=min_start,
        max_start=max_start,
        min_end=min_end,
        max_end=max_end,
    )
    return annual, q1_to_q3


def find_sec_diluted_shares(row: dict, shares: pd.DataFrame) -> int:
    end = pd.Timestamp(row['date'])
    fallback = row['weightedAverageShsOutDil']
    if row['period'] != 'Q4':
        datum = select_sec_quarter_datum(
            shares,
            description=f'CIK{row["cik"]} {date_str(end)}',
            end=end,
        )
        if datum is None:
            return fallback
        return datum['val']

    annual, q1_to_q3 = select_sec_q4_components(
        shares,
        annual_description=f'CIK{row["cik"]} annual {date_str(end)}',
        q1_to_q3_description_prefix=f'CIK{row["cik"]} Q1-Q3 from ',
        end=end,
    )
    if annual is None or q1_to_q3 is None:
        return fallback
    return round(annual['val'] * 4 - q1_to_q3['val'] * 3)


def find_sec_diluted_eps(row: dict, eps: pd.DataFrame, eps_col: str) -> float:
    end = pd.Timestamp(row['date'])
    fallback = row[eps_col]
    if row['period'] != 'Q4':
        datum = select_sec_quarter_datum(
            eps,
            description=f'CIK{row["cik"]} EPS {date_str(end)}',
            end=end,
        )
        if datum is None:
            return fallback
        return datum['val']

    annual, q1_to_q3 = select_sec_q4_components(
        eps,
        annual_description=f'CIK{row["cik"]} EPS annual {date_str(end)}',
        q1_to_q3_description_prefix=f'CIK{row["cik"]} EPS Q1-Q3 from ',
        end=end,
    )
    if annual is None or q1_to_q3 is None:
        return fallback
    return annual['val'] - q1_to_q3['val']


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
        raise
    return dedupe_sec_rows(data['units']['shares'])


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
        raise
    return dedupe_sec_rows(data['units']['USD/shares'])


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


@cached(240)
async def fetch_xps(
    market: Literal['j', 't', 'u'], symbol: str, q: int
) -> pd.DataFrame:
    limit = q + 3
    params = {
        'apikey': FMP_KEY,
        'limit': limit,
        'period': 'quarter',
    }
    if market == 'u':
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
        eps_col = 'epsDiluted'
    else:
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ add_suffix(market, symbol) }'
        eps_col = 'epsdiluted'
    async with AsyncClient() as client:
        data = (await client.get(url, params=params)).json()
    path = PATCH_DIR / f'{symbol}.json'
    if path.exists():
        with path.open() as f:
            patch = json.load(f)
        data = sorted(
            {r['date']: r for r in data + patch}.values(),
            key=lambda r: r['date'],
            reverse=True,
        )[:limit]
    if len(data) != limit:
        raise ValueError
    for i, row in enumerate(data):
        if (
            market == 'u'
            and row.get('reportedCurrency') == 'USD'
            and (diff := share_count_log_diff(row, eps_col))
            > SHARE_COUNT_DIFF_THRESHOLD
        ):
            logger.warning(
                'Income statement inconsistency detected: {} FY{} {} diff={:.4f}',
                row.get('symbol', symbol),
                row.get('fiscalYear'),
                row.get('period'),
                diff,
            )
            share_fix, eps_fix = await asyncio.gather(
                verify_income_statement_diluted_shares(row),
                verify_income_statement_eps(row, eps_col),
            )
            data[i] = row | share_fix | eps_fix
            cleaned_diff = share_count_log_diff(data[i], eps_col)
            logger.info(
                'Income statement repaired from SEC data: {} FY{} {} diff={:.4f}',
                data[i].get('symbol', symbol),
                data[i].get('fiscalYear'),
                data[i].get('period'),
                cleaned_diff,
            )
    df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
    return pd.DataFrame(
        {
            'rps': (df['revenue'] / df['weightedAverageShsOutDil'])
            .rolling(4)
            .sum()
            .to_numpy(),
            'eps': df[eps_col].rolling(4).sum().to_numpy(),
        },
        pd.to_datetime(df['date']) + pd.Timedelta(days=1),
    ).iloc[3:]


async def calc_scores(
    market: Literal['j', 't', 'u'], symbol: str, end_date: str, q: int, ema7: bool
) -> tuple[float, float | None]:
    prices, xps = await asyncio.gather(
        shared.get_prices(market, symbol, 91 * q, False, ema7),
        fetch_xps(market, symbol, q + EXTRA_Q),
    )
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(prices),
    )
    e = pd.Timestamp(end_date)
    s = e - pd.Timedelta(days=91 * q - 1)
    df = pd.DataFrame({'price': prices}, index).join(xps, how='outer').ffill().loc[s:e]
    if (
        (len(df) != 91 * q)
        or (pd.isna(df['price'].iloc[0]))
        or pd.isna(df['rps'].iloc[0])
    ):
        raise ValueError

    def percentile_rank(s: pd.Series) -> float:
        return (s < s.iloc[-1]).mean()

    return (
        percentile_rank(df['price'] / df['rps']),
        percentile_rank(df['price'] / df['eps']) if (df['eps'] > 0).all() else None,
    )


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
