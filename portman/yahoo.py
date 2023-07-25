import asyncio
from typing import Literal

import arrow
from httpx import AsyncClient

from .shared import (
    MARKET_TO_TIMEZONE,
    clean_up,
    gen_dates,
    get_sorted_values,
    get_today_dividend,
)


async def get_unadjusted(
    h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int
):
    tz = MARKET_TO_TIMEZONE[market]
    now = arrow.now(tz)
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-(n + 13)), now), 0.0)
    res = await h.get(
        f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}{ ".TW" if market == "t" else "" }',
        params={
            'events': 'history',
            'period1': int(arrow.get(min(date_to_price), tzinfo=tz).timestamp()),
            'period2': int(
                arrow.get(max(date_to_price), tzinfo=tz).shift(days=1).timestamp()
            ),
        },
    )
    for l in res.text.split('\n')[1:]:
        if len(l := l.split(',')) > 1:
            if l[0] in date_to_price:
                date_to_price[l[0]] = float(l[4])
    return clean_up(get_sorted_values(date_to_price), n)


async def get_dividends(h: AsyncClient, market: Literal['t', 'u'], symbol: str, n: int):
    tz = MARKET_TO_TIMEZONE[market]
    now = arrow.now(tz)
    date_to_dividend = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    res, today = await asyncio.gather(
        h.get(
            f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}{ ".TW" if market == "t" else "" }',
            params={
                'events': 'div',
                'period1': int(arrow.get(min(date_to_dividend), tzinfo=tz).timestamp()),
                'period2': int(
                    arrow.get(max(date_to_dividend), tzinfo=tz)
                    .shift(days=1)
                    .timestamp()
                ),
            },
        ),
        get_today_dividend(h, market, symbol),
    )
    for l in res.text.split('\n')[1:]:
        if len(l := l.split(',')) > 1:
            if l[0] in date_to_dividend:
                date_to_dividend[l[0]] = float(l[1])
    date_to_dividend[max(date_to_dividend)] = today
    return get_sorted_values(date_to_dividend)[-n:]
