import asyncio
from typing import Literal

import arrow
import numba as nb
import numpy as np
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import yahoo
from .shared import (
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
    return post_process(get_sorted_values(date_to_price), n, market == 't')


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
            'from': min(date_to_rate),
            'serietype': 'line',
        },
    )
    for e in res.json()['historical']:
        if e['date'] in date_to_rate:
            date_to_rate[e['date']] = e['close']
    return post_process(get_sorted_values(date_to_rate), n)


async def get_prices(
    market: Literal['t', 'u'], symbol: str, n: int, to_usd: bool = True
):
    async with AsyncClient(timeout=60, params={'apikey': FMP_KEY}) as sess:
        tasks = [get_adjusted(sess, market, symbol, n)]
        if to_usd:
            tasks.append(get_rates(sess, n))
        results = await asyncio.gather(*tasks)
    return results[0] / results[1] if to_usd else results[0]
