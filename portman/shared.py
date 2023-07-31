import json
import os
import re
from typing import Literal

import aiofiles
import arrow
import dotenv
import numba as nb
import numpy as np
from arrow.arrow import Arrow
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

dotenv.load_dotenv()

FMP_KEY = os.environ['FMP_KEY']
FROM_GECKO = set(os.environ['FROM_GECKO'].split(','))
FROM_YAHOO = set(os.environ['FROM_YAHOO'].split(','))
MARKET_TO_TIMEZONE = {'c': 'UTC', 't': 'Asia/Taipei', 'u': 'America/New_York'}


def to_date(time: Arrow | int | str, timezone: str = ''):
    if not isinstance(time, Arrow):
        time = arrow.get(time)
    if timezone:
        time = time.to(timezone)
    return time.format('YYYY-MM-DD')


def gen_dates(start: Arrow, end: Arrow):
    return map(to_date, Arrow.range('day', start, end))


def get_sorted_values(D: dict[str, float]):
    return np.array([e[1] for e in sorted(D.items(), key=lambda x: x[0])])


@nb.njit
def clean_up(arr: Array[f8], n: int):
    arr = arr.copy()  # pyright: ignore

    def diff(a: f8, b: f8):
        return np.abs(np.log(a / b))

    prev_diff = 0
    for i in range(1, len(arr)):
        if not arr[i]:
            arr[i] = arr[i - 1]
            prev_diff = 0
        elif arr[i - 1]:
            curr_diff = diff(arr[i - 1], arr[i])
            if prev_diff > 0.693 and curr_diff > 0.693:
                arr[i - 1] = arr[i - 2]
            prev_diff = curr_diff

    arr = arr[-n:]  # pyright: ignore
    assert arr[0]
    return arr


async def get_today_dividend(h: AsyncClient, market: Literal['t', 'u'], symbol: str):
    today = to_date(arrow.now(MARKET_TO_TIMEZONE[market]))
    if market == 't':
        async with aiofiles.open('TWT48U.json') as f:
            text = await f.read()
        for e in json.loads(text)['data']:
            y, m, d = map(int, re.findall('\\d+', e[0]))
            if to_date(Arrow(y + 1911, m, d)) == today and e[1] == symbol:
                return float(e[7])
    else:
        res = await h.get(
            'https://financialmodelingprep.com/api/v3/stock_dividend_calendar',
            params={
                'from': today,
                'to': today,
            },
        )
        for e in res.json():
            if e['date'] == today and e['symbol'] == symbol:
                return e['dividend']
    return 0.0
