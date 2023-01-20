from shared import *

cache = {}


@dataclass
class CachedResponse:
    hit: bool
    text: str

    def json(self) -> dict:
        return json.loads(self.text)


async def cached_get(h: AsyncClient, url: str) -> CachedResponse:
    if not (hit := url in cache):
        logging.info(url)
        res = await h.get(url)
        res.raise_for_status()
        cache[url] = res.text
    return CachedResponse(hit, cache[url])


async def get_indices(h: AsyncClient, n: int) -> Array[f8]:
    global cache
    if os.path.exists('.TW50.json'):
        with open('.TW50.json', 'r') as f:
            cache = json.load(f)
        del cache[max(cache)]
    now = arrow.now(MARKET_TO_TIMEZONE['t'])
    start = now.shift(days=-(n + 13))
    date_to_price = dict.fromkeys(gen_dates(start, now), 0.0)
    for a in Arrow.range('month', start.replace(day=1), now):
        try:
            if not res.hit:  # type: ignore
                await asyncio.sleep(4)
        except NameError:
            pass
        res = await cached_get(
            h,
            f'https://www.twse.com.tw/indicesReport/TAI50I?date={ a.format("YYYYMMDD") }',
        )
        for e in res.json()['data']:
            y, m, d = map(int, e[0].split('/'))
            if (date := to_date(Arrow(y + 1911, m, d))) in date_to_price:
                date_to_price[date] = float(e[2].replace(',', ''))
    with open('.TW50.json', 'w') as f:
        json.dump(cache, f)
    return clean_up(get_values(date_to_price))[-n:]
