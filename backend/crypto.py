import arrow
from httpx import AsyncClient

from .shared import (
    FROM_COINGECKO,
    MARKET_TO_TIMEZONE,
    gen_dates,
    get_sorted_values,
    post_process,
    to_date,
)


async def get_id(sess: AsyncClient, symbol: str):
    res = await sess.get(
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
    async with AsyncClient(timeout=60) as sess:
        if symbol in FROM_COINGECKO:
            res = await sess.get(
                f'https://api.coingecko.com/api/v3/coins/{await get_id(sess, symbol)}/market_chart',
                params={
                    'days': len(date_to_price),
                    'vs_currency': 'usd',
                },
            )
            for e in res.json()['prices']:
                if (date := to_date(e[0], tz)) in date_to_price:
                    date_to_price[date] = e[1]
        else:
            # Binance caps klines `limit` at 1000, so page backwards for long windows.
            end_time = None
            while True:
                params = {
                    'interval': '1d',
                    'limit': 1000,
                    'symbol': symbol + 'USDT',
                }
                if end_time is not None:
                    params['endTime'] = end_time
                res = await sess.get('https://api1.binance.com/api/v3/klines', params=params)
                res.raise_for_status()
                klines = res.json()
                if not klines:
                    break
                for e in klines:
                    if (date := to_date(e[0], tz)) in date_to_price:
                        date_to_price[date] = float(e[4])
                if to_date(klines[0][0], tz) <= min(date_to_price) or len(klines) < 1000:
                    break
                end_time = klines[0][0] - 1
    try:
        return post_process(get_sorted_values(date_to_price), n)
    except AssertionError:
        raise AssertionError(symbol)
