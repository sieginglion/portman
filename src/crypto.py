import arrow
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from shared import clean_up, gen_dates, get_values, to_date


async def get_prices(symbol: str, n: int) -> Array[f8]:
    now = arrow.now('UTC')
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    async with AsyncClient(http2=True) as h:
        res = await h.get(
            f'https://api.binance.com/api/v3/klines?symbol={symbol}BUSD&interval=1d&limit={n}'
        )
    for e in res.json():
        if (date := to_date(e[0], 'UTC')) in date_to_price:
            date_to_price[date] = e[4]
    return clean_up(get_values(date_to_price))[-n:]
