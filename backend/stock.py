import asyncio
from typing import Literal

import arrow
import numba as nb
import numpy as np
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from .shared import (
    FMP_KEY,
    MARKET_TO_TIMEZONE,
    add_suffix,
    gen_dates,
    get_sorted_values,
    get_today_dividend,
    post_process,
)


def daily_values(start, end, historical, value_key):
    date_to_value = dict.fromkeys(gen_dates(start, end), 0.0)
    for entry in historical:
        if entry['date'] in date_to_value:
            date_to_value[entry['date']] = entry[value_key]
    return date_to_value


async def _fetch_fmp_price_history(sess: AsyncClient, symbol: str, start):
    res = await sess.get(
        f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}',
        params={
            'from': start.format('YYYY-MM-DD'),
            'serietype': 'line',
        },
    )
    return res.json()['historical']


def _normalize_fmp_price_history(historical, start, end, n: int, limited: bool):
    date_to_value = daily_values(start, end, historical, 'close')
    return post_process(get_sorted_values(date_to_value), n, limited)


async def get_fmp_series(
    sess: AsyncClient, symbol: str, start, end, n: int, limited: bool = False
):
    historical = await _fetch_fmp_price_history(sess, symbol, start)
    return _normalize_fmp_price_history(historical, start, end, n, limited)


def _market_window(market: Literal['j', 't', 'u'], n: int):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    return now.shift(days=-(n + 13)), now


async def get_fmp_unadjusted(
    sess: AsyncClient, market: Literal['j', 't', 'u'], symbol: str, n: int
):
    start, now = _market_window(market, n)
    try:
        return await get_fmp_series(
            sess, add_suffix(market, symbol), start, now, n, market == 't'
        )
    except AssertionError:
        raise AssertionError(symbol)


async def get_fmp_dividends(
    sess: AsyncClient, market: Literal['j', 't', 'u'], symbol: str, n: int
):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    start = now.shift(days=-n)
    res, today = await asyncio.gather(
        sess.get(
            f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ add_suffix(market, symbol) }'
        ),
        get_today_dividend(market, symbol),
    )
    date_to_dividend = daily_values(
        start, now, res.json().get('historical', []), 'adjDividend'
    )
    date_to_dividend[max(date_to_dividend)] = today
    return get_sorted_values(date_to_dividend)[-n:]


@nb.njit
def calc_adjusted(unadjusted: Array[f8], dividends: Array[f8]):
    n = len(dividends)
    adjusted = np.empty(n)
    factor = 1.0
    for i in range(n - 1, 0, -1):
        adjusted[i] = unadjusted[i] * factor
        if dividends[i]:
            factor *= 1 - dividends[i] / unadjusted[i - 1]
    adjusted[0] = unadjusted[0] * factor
    return adjusted


async def get_adjusted(
    sess: AsyncClient, market: Literal['j', 't', 'u'], symbol: str, n: int
):
    unadjusted, dividends = await asyncio.gather(
        get_fmp_unadjusted(sess, market, symbol, n),
        get_fmp_dividends(sess, market, symbol, n),
    )
    return calc_adjusted(unadjusted, dividends)


async def get_rates(sess: AsyncClient, n: int):
    start, now = _market_window('t', n)
    return await get_fmp_series(sess, 'USDTWD', start, now, n)


async def get_prices(market: Literal['j', 't', 'u'], symbol: str, n: int, to_usd: bool):
    async with AsyncClient(timeout=60, params={'apikey': FMP_KEY}) as sess:
        if market == 't' and to_usd:
            prices, rates = await asyncio.gather(
                get_adjusted(sess, market, symbol, n), get_rates(sess, n)
            )
            return prices / rates
        return await get_adjusted(sess, market, symbol, n)
