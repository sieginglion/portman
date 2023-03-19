import asyncio
import re
from typing import Literal

import arrow
import numba as nb
import numpy as np
from arrow import Arrow
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from shared import (FMP_KEY, MARKET_TO_TIMEZONE, cached_get, clean_up,
                    gen_dates, get_values, to_date)


async def get_unadjusted(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
) -> Array[f8]:
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
    return clean_up(get_values(date_to_price))[-n:]


async def get_today_dividend(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str
) -> float:
    today = to_date(arrow.now(MARKET_TO_TIMEZONE[market]))
    if market == 't':
        res = cached_get('https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json')
        for e in res.json()['data']:
            y, m, d = map(int, re.findall('\\d+', e[0]))
            if to_date(Arrow(y + 1911, m, d)) == today and e[1] == symbol:
                return float(e[7])
    else:
        res = await h.get(
            'https://financialmodelingprep.com/api/v3/stock_dividend_calendar',
            params={
                'from': today,
                'to': today,
            },
        )
        for e in res.json():
            if e['date'] == today and e['symbol'] == symbol:
                return e['dividend']
    return 0.0


async def get_dividends(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
) -> Array[f8]:
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
    return get_values(date_to_dividend)[-n:]


@nb.njit
def calc_adjusted(unadjusted: Array[f8], dividends: Array[f8]) -> Array[f8]:
    A = np.ones(len(dividends))
    for i, dividend in enumerate(dividends, 1):
        if dividend:
            A[i - 1] = 1 - dividend / unadjusted[i - 1]
    return unadjusted * np.flip(np.cumprod(np.flip(A)))


async def get_adjusted(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
) -> Array[f8]:
    unadjusted, dividends = await asyncio.gather(
        get_unadjusted(h, market, symbol, n), get_dividends(h, market, symbol, n)
    )
    return calc_adjusted(unadjusted, dividends)


async def get_rates(h: AsyncClient, market: Literal['t', 'u'], n: int) -> Array[f8]:
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
    return clean_up(get_values(date_to_rate))[-n:]


async def get_prices(market: Literal['t', 'u'], symbol: str, n: int) -> Array[f8]:
    async with AsyncClient(http2=True, params={'apikey': FMP_KEY}) as h:
        adjusted, rates = await asyncio.gather(
            get_adjusted(h, market, symbol, n), get_rates(h, market, n)
        )
    return np.trim_zeros(adjusted / rates, 'f')
