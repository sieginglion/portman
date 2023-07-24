import arrow
from httpx import AsyncClient

from .shared import MARKET_TO_TIMEZONE, clean_up, gen_dates, get_sorted_values, to_date


async def get_prices(symbol: str, n: int):
    now = arrow.now(MARKET_TO_TIMEZONE['c'])
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    async with AsyncClient() as h:
        res = await h.get(
            f'https://api.binance.com/api/v3/klines?symbol={symbol}BUSD&interval=1d&limit={n}'
        )
    for e in res.json():
        if (date := to_date(e[0], MARKET_TO_TIMEZONE['c'])) in date_to_price:
            date_to_price[date] = float(e[4])
    return clean_up(get_sorted_values(date_to_price), n)
