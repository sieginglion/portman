from shared import *

symbol_to_id = {
    e['symbol'].upper(): e['id']
    for e in r.get(
        'https://api.coingecko.com/api/v3/coins/markets',
        params={
            'per_page': 100,
            'vs_currency': 'usd',
        },
        timeout=10,
    ).json()
}


async def get_prices(symbol: str, n: int) -> Array[f8]:
    now = arrow.now('UTC')
    date_to_price = dict.fromkeys(gen_dates(now.shift(days=-n), now), 0.0)
    async with AsyncClient(http2=True) as h:
        res = await h.get(
            f'https://api.coingecko.com/api/v3/coins/{ symbol_to_id[symbol] }/market_chart',
            params={
                'days': n + 1,
                'interval': 'daily',
                'vs_currency': 'usd',
            },
        )
    for e in res.json()['prices']:
        if (date := to_date(e[0], 'UTC')) in date_to_price:
            date_to_price[date] = e[1]
    return clean_up(get_values(date_to_price))[-n:]
