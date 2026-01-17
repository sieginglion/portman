import asyncio
from typing import Literal

import arrow
import numba as nb
import numpy as np
import pandas as pd
from general_cache import cached
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import yahoo
from .shared import (
    FINMIND_KEY,
    FMP_KEY,
    FROM_YAHOO,
    MARKET_TO_TIMEZONE,
    gen_dates,
    get_sorted_values,
    get_suffix,
    get_today_dividend,
    post_process,
)


async def get_unadjusted(
    sess: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    historical, quote = map(
        lambda x: x.json(),
        await asyncio.gather(
            sess.get(
                f'https://financialmodelingprep.com/api/v3/historical-price-full/{ symbol }{ get_suffix(market, symbol) }',
                params={
                    'from': min(date_to_price),
                    'serietype': 'line',
                },
            ),
            sess.get(
                f'https://financialmodelingprep.com/api/v3/quote-short/{ symbol }{ get_suffix(market, symbol) }'
            ),
        ),
    )
    for e in historical['historical']:
        if e['date'] in date_to_price:
            date_to_price[e['date']] = e['close']
    date_to_price[max(date_to_price)] = quote[0]['price']
    try:
        return post_process(get_sorted_values(date_to_price), n, market == 't')
    except AssertionError:
        raise AssertionError(symbol)


async def get_dividends(
    sess: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    date_to_dividend = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    res, today = await asyncio.gather(
        sess.get(
            f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ symbol }{ get_suffix(market, symbol) }'
        ),
        get_today_dividend(sess, market, symbol),
    )
    for e in res.json().get('historical', []):
        if e['date'] in date_to_dividend:
            date_to_dividend[e['date']] = e['adjDividend']  # to avoid stock splits
    date_to_dividend[max(date_to_dividend)] = today
    return get_sorted_values(date_to_dividend)[-n:]


@nb.njit
def calc_adjusted(unadjusted: Array[f8], dividends: Array[f8]):
    n = len(dividends)
    adjusted = np.empty(n)
    factor = 1
    for i in range(n - 1, -1, -1):
        adjusted[i] = unadjusted[i] * factor
        if dividends[i]:
            factor *= 1 - dividends[i] / unadjusted[i - 1]
    return adjusted


async def get_adjusted(
    sess: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    args = (sess, market, symbol, n)
    funcs = (
        (yahoo.get_unadjusted(*args), yahoo.get_dividends(*args))
        if symbol in FROM_YAHOO
        else (get_unadjusted(*args), get_dividends(*args))
    )
    unadjusted, dividends = await asyncio.gather(*funcs)
    return calc_adjusted(unadjusted, dividends)


async def get_rates(sess: AsyncClient, n: int):
    now = arrow.now(MARKET_TO_TIMEZONE['t'])
    date_to_rate = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    res = await sess.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/USDTWD',
        params={
            'apikey': FMP_KEY,
            'from': min(date_to_rate),
            'serietype': 'line',
        },
    )
    for e in res.json()['historical']:
        if e['date'] in date_to_rate:
            date_to_rate[e['date']] = e['close']
    return post_process(get_sorted_values(date_to_rate), n)


@cached(240)
async def get_prices(
    market: Literal['t', 'u'], symbol: str, n: int, to_usd: bool = True
):
    async with AsyncClient(timeout=60, params={'apikey': FMP_KEY}) as sess:
        tasks = [get_adjusted(sess, market, symbol, n)]
        if to_usd:
            tasks.append(get_rates(sess, n))
        results = await asyncio.gather(*tasks)
    return results[0] / results[1] if to_usd else results[0]


async def calc_rps_from_fmp(symbol: str):
    url = 'https://financialmodelingprep.com/stable/income-statement'
    params = {
        'apikey': FMP_KEY,
        'limit': 8,
        'period': 'quarter',
        'symbol': symbol,
    }
    async with AsyncClient() as client:
        d = (await client.get(url, params=params)).json()
    if len(d) != 8:
        raise AssertionError
    df = pd.DataFrame(d).set_index('date').sort_index()
    df['rps'] = (df['revenue'] / df['weightedAverageShsOutDil']).rolling(4).sum()
    return df['rps'].iloc[3:]


def avg_by_period_ends(values, ends):
    today = pd.Timestamp.now(tz='Asia/Taipei').normalize().tz_localize(None)
    series = pd.Series(
        values, index=pd.date_range(end=today, periods=len(values), freq='D')
    )
    ends = pd.to_datetime(ends)
    starts = ends[:-1] + pd.Timedelta(days=1)
    return pd.Series({e: series.loc[s:e].mean() for s, e in zip(starts, ends[1:])})


async def calc_rps_from_finmind(symbol: str):
    url = 'https://api.finmindtrade.com/api/v4/data'
    params = {
        'data_id': symbol,
        'dataset': 'TaiwanStockFinancialStatements',
        'start_date': arrow.now('Asia/Taipei')
        .shift(days=-10 * 91)
        .format('YYYY-MM-DD'),
    }
    headers = {'Authorization': f'Bearer {FINMIND_KEY}'}
    async with AsyncClient() as client:
        resp, fx = await asyncio.gather(
            client.get(url, params=params, headers=headers), get_rates(client, 10 * 91)
        )
    df = (
        pd.DataFrame(resp.json()['data'])
        .pivot(index='date', columns='type', values='value')
        .sort_index()
        .iloc[-9:]
    )
    if len(df) != 9:
        raise AssertionError
    fx = avg_by_period_ends(fx, df.index)
    df = df.iloc[1:]
    df['Revenue'] /= fx.values
    df['rps'] = (
        (df['Revenue'] * df['EPS'] / df['EquityAttributableToOwnersOfParent'])
        .rolling(4)
        .sum()
    )
    return df['rps'].iloc[3:]


# async def calc_bands(market: Literal['t', 'u'], symbol: str):
#     prices = await get_prices(market, symbol, 364)
#     rps = await calc_rps_from_finmind(symbol)
#     dates = pd.date_range(incomes[0].d, prices.index[-1]).date
#     s = (
#         pd.Series({income.d: getattr(income, metric) for income in incomes}, dates)
#         .ffill()
#         .tail(len(prices))
#     )
#     s[s <= 0] = None
#     multiples = prices / s
#     min_m, max_m = multiples.quantile(0.02), multiples.quantile(0.98)
#     bands = pd.DataFrame(index=s.index)
#     if not min_m < max_m:
#         return bands
#     for p in range(0, 120, 20):
#         m = min_m + (max_m - min_m) * (p / 100)
#         bands[m] = s * m
#     return bands
