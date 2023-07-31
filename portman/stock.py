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
    clean_up,
    gen_dates,
    get_sorted_values,
    get_today_dividend,
)


async def get_unadjusted(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    historical, quote = map(
        lambda x: x.json(),
        await asyncio.gather(
            h.get(
                f'https://financialmodelingprep.com/api/v3/historical-price-full/{ symbol }{ ".TW" if market == "t" else "" }',
                params={
                    'from': min(date_to_price),
                    'serietype': 'line',
                },
            ),
            h.get(
                f'https://financialmodelingprep.com/api/v3/quote-short/{ symbol }{ ".TW" if market == "t" else "" }'
            ),
        ),
    )
    for e in historical['historical']:
        if e['date'] in date_to_price:
            date_to_price[e['date']] = e['close']
    date_to_price[max(date_to_price)] = quote[0]['price']
    return clean_up(get_sorted_values(date_to_price), n)


async def get_dividends(h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    date_to_dividend = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    res, today = await asyncio.gather(
        h.get(
            f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ symbol }{ ".TW" if market == "t" else "" }'
        ),
        get_today_dividend(h, market, symbol),
    )
    for e in res.json().get('historical', []):
        if e['date'] in date_to_dividend:
            date_to_dividend[e['date']] = e['adjDividend']  # to avoid stock splits
    date_to_dividend[max(date_to_dividend)] = today
    return get_sorted_values(date_to_dividend)[-n:]


@nb.njit
def calc_adjusted(unadjusted: Array[f8], dividends: Array[f8]):
    factors = np.ones(len(dividends))
    for i, dividend in enumerate(dividends, 1):
        if dividend:
            factors[i - 1] = 1 - dividend / unadjusted[i - 1]
    return unadjusted * np.flip(np.cumprod(np.flip(factors)))


async def get_adjusted(h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int):
    args = (h, market, symbol, n)
    funcs = (
        (yahoo.get_unadjusted(*args), yahoo.get_dividends(*args))
        if symbol in FROM_YAHOO
        else (get_unadjusted(*args), get_dividends(*args))
    )
    unadjusted, dividends = await asyncio.gather(*funcs)
    return calc_adjusted(unadjusted, dividends)


async def get_rates(h: AsyncClient, market: Literal['t', 'u'], n: int):
    if market == 'u':
        return np.ones(n)
    now = arrow.now(MARKET_TO_TIMEZONE['t'])
    date_to_rate = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    res = await h.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/USDTWD',
        params={
            'from': min(date_to_rate),
            'serietype': 'line',
        },
    )
    for e in res.json()['historical']:
        if e['date'] in date_to_rate:
            date_to_rate[e['date']] = e['close']
    return clean_up(get_sorted_values(date_to_rate), n)


async def get_prices(market: Literal['t', 'u'], symbol: str, n: int):
    async with AsyncClient(params={'apikey': FMP_KEY}) as h:
        adjusted, rates = await asyncio.gather(
            get_adjusted(h, market, symbol, n), get_rates(h, market, n)
        )
    return adjusted / rates
