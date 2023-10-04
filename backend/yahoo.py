import asyncio
from typing import Literal

import arrow
from httpx import AsyncClient

from .shared import (
    MARKET_TO_TIMEZONE,
    gen_dates,
    get_sorted_values,
    get_suffix,
    get_today_dividend,
    patch_and_trunc,
)


async def get_unadjusted(
    sess: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    tz = MARKET_TO_TIMEZONE[market]
    now = arrow.now(tz)
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    res = await sess.get(
        f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}{ get_suffix(market, symbol) }',
        params={
            'events': 'history',
            'period1': int(arrow.get(min(date_to_price), tzinfo=tz).timestamp()),
            'period2': int(arrow.get(max(date_to_price), tzinfo=tz).timestamp())
            + 86400,
        },
    )
    for l in res.text.split('\n')[1:]:
        if len(l := l.split(',')) > 5:
            if l[0] in date_to_price:
                date_to_price[l[0]] = float(l[4])
    return patch_and_trunc(get_sorted_values(date_to_price), n)


async def get_dividends(
    sess: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    tz = MARKET_TO_TIMEZONE[market]
    now = arrow.now(tz)
    date_to_dividend = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    res, today = await asyncio.gather(
        sess.get(
            f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}{ get_suffix(market, symbol) }',
            params={
                'events': 'div',
                'period1': int(arrow.get(min(date_to_dividend), tzinfo=tz).timestamp()),
                'period2': int(arrow.get(max(date_to_dividend), tzinfo=tz).timestamp())
                + 86400,
            },
        ),
        get_today_dividend(sess, market, symbol),
    )
    for l in res.text.split('\n')[1:]:
        if len(l := l.split(',')) > 1:
            if l[0] in date_to_dividend:
                date_to_dividend[l[0]] = float(l[1])
    date_to_dividend[max(date_to_dividend)] = today
    return get_sorted_values(date_to_dividend)[-n:]
