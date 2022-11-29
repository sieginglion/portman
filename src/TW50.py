from shared import *

try:
    with open('.TW50.pkl', 'rb') as f:
        cache = pickle.load(f)
except FileNotFoundError:
    cache = {}


async def get_patch(start: Arrow, end: Arrow) -> Dict[str, f8]:
    n = (end - start).days + 1
    start = start.shift(days=-14)
    date_to_price = dict.fromkeys(gen_dates(start, end), 0.0)
    async with AsyncClient(http2=True) as h:
        for i, a in enumerate(Arrow.range('month', start, end)):
            if i:
                await asyncio.sleep(4)
            res = await h.get(
                f'https://www.twse.com.tw/indicesReport/TAI50I?date={ a.format("YYYYMMDD") }'
            )
            print(res.url)
            for e in res.json()['data']:
                y, m, d = map(int, e[0].split('/'))
                date = to_date(Arrow(y + 1911, m, d))
                if date in date_to_price:
                    date_to_price[date] = float(e[1].replace(',', ''))
    dates, prices = zip(*sorted(date_to_price.items(), key=lambda x: x[0]))
    return dict(zip(dates[-n:], get_patched(np.array(prices))[-n:]))  # type: ignore


async def get_indices(n: int) -> A[f8]:
    now = arrow.now(market_to_timezone['t'])
    dates = gen_dates(now.shift(days=-(n - 1)), now)
    missing = sorted(set(dates) - cache.keys())
    if missing:
        cache.update(await get_patch(arrow.get(min(missing)), arrow.get(max(missing))))
        with open('.TW50.pkl', 'wb') as f:
            pickle.dump(cache, f)
    return np.array(operator.itemgetter(*dates)(cache))


# asyncio.run(get_indices(32))
