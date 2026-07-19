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


async def get_unadjusted(
    sess: AsyncClient, market: Literal['j', 't', 'u'], symbol: str, n: int
):
    now = arrow.now(MARKET_TO_TIMEZONE[market])
    start = now.shift(days=-(n + 13))
    # historical, quote = map(
    #     lambda x: x.json(),
    #     await asyncio.gather(
    #         sess.get(
    #             f'https://financialmodelingprep.com/api/v3/historical-price-full/{ add_suffix(market, symbol) }',
    #             params={
    #                 'from': min(date_to_price),
    #                 'serietype': 'line',
    #             },
    #         ),
    #         sess.get(
    #             f'https://financialmodelingprep.com/api/v3/quote-short/{ add_suffix(market, symbol) }'
    #         ),
    #     ),
    # )
    historical = (
        await sess.get(
            f'https://financialmodelingprep.com/api/v3/historical-price-full/{ add_suffix(market, symbol) }',
            params={
                'from': start.format('YYYY-MM-DD'),
                'serietype': 'line',
            },
        )
    ).json()
    date_to_price = daily_values(start, now, historical['historical'], 'close')
    # date_to_price[max(date_to_price)] = quote[0]['price']
    try:
        return post_process(get_sorted_values(date_to_price), n, market == 't')
    except AssertionError:
        raise AssertionError(symbol)


async def get_dividends(
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
    start = now.shift(days=-(n + 13))
    res = await sess.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/USDTWD',
        params={
            'apikey': FMP_KEY,
            'from': start.format('YYYY-MM-DD'),
            'serietype': 'line',
        },
    )
    date_to_rate = daily_values(start, now, res.json()['historical'], 'close')
    return post_process(get_sorted_values(date_to_rate), n)


async def get_prices(market: Literal['j', 't', 'u'], symbol: str, n: int, to_usd: bool):
    async with AsyncClient(timeout=60, params={'apikey': FMP_KEY}) as sess:
        if market == 't' and to_usd:
            prices, rates = await asyncio.gather(
                get_adjusted(sess, market, symbol, n), get_rates(sess, n)
            )
            return prices / rates
        return await get_adjusted(sess, market, symbol, n)
