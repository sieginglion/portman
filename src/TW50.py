from shared import *

try:
    with open('.TW50.json', 'r') as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}


@dataclass
class CachedResponse:
    hit: bool
    text: str

    def json(self):
        return json.loads(self.text)


async def cached_get(url: str) -> CachedResponse:
    if not (hit := url in cache):
        async with AsyncClient(http2=True) as h:
            res = await h.get(url)
        print(url)
        res.raise_for_status()
        cache[url] = res.text
        with open('.TW50.json', 'w') as f:
            json.dump(cache, f)
    return CachedResponse(hit, cache[url])


async def get_indices(n: int) -> A[f8]:
    now = arrow.now(market_to_timezone['t'])
    start = now.shift(days=-(n + 13))
    date_to_price = dict.fromkeys(gen_dates(start, now), 0.0)
    for a in Arrow.range('month', start.replace(day=1), now):
        try:
            if not res.hit:  # type: ignore
                await asyncio.sleep(4)
        except NameError:
            pass
        res = await cached_get(
            f'https://www.twse.com.tw/indicesReport/TAI50I?date={ a.format("YYYYMMDD") }'
        )
        for e in res.json()['data']:
            y, m, d = map(int, e[0].split('/'))
            if (date := to_date(Arrow(y + 1911, m, d))) in date_to_price:
                date_to_price[date] = float(e[1].replace(',', ''))
    return get_patched(get_values(date_to_price))[-n:]


# asyncio.run(get_indices(32))
