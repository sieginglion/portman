import arrow
from httpx import AsyncClient

from .shared import (
    FROM_GECKO,
    MARKET_TO_TIMEZONE,
    clean_up,
    gen_dates,
    get_sorted_values,
    to_date,
)


async def get_id(h: AsyncClient, symbol: str):
    res = await h.get(
        'https://api.coingecko.com/api/v3/coins/markets',
        params={
            'per_page': 200,
            'vs_currency': 'usd',
        },
    )
    for e in res.json():
        if e['symbol'].upper() == symbol:
            return e['id']


async def get_prices(symbol: str, n: int):
    tz = MARKET_TO_TIMEZONE['c']
    now = arrow.now(tz)
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    async with AsyncClient(timeout=60) as h:
        if symbol in FROM_GECKO:
            res = await h.get(
                f'https://api.coingecko.com/api/v3/coins/{await get_id(h, symbol)}/market_chart',
                params={
                    'days': len(date_to_price),
                    'vs_currency': 'usd',
                },
            )
            for e in res.json()['prices']:
                if (date := to_date(e[0], tz)) in date_to_price:
                    date_to_price[date] = e[1]
        else:
            res = await h.get(
                'https://api1.binance.com/api/v3/klines',
                params={
                    'interval': '1d',
                    'limit': len(date_to_price),
                    'symbol': symbol + 'USDT',
                },
            )
            for e in res.json():
                if (date := to_date(e[0], tz)) in date_to_price:
                    date_to_price[date] = float(e[4])
    return clean_up(get_sorted_values(date_to_price), n)
