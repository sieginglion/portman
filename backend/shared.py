import json
import os
import re
from typing import Literal

import arrow
import dotenv
import numba as nb
import numpy as np
import requests as r
from arrow.arrow import Arrow
from general_cache import cached
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

dotenv.load_dotenv()

FMP_KEY = os.environ['FMP_KEY']
FROM_COINGECKO = set(os.environ['FROM_COINGECKO'].split(','))
FROM_YAHOO = set(os.environ['FROM_YAHOO'].split(','))
MARKET_TO_TIMEZONE = {'c': 'UTC', 't': 'Asia/Taipei', 'u': 'America/New_York'}
ON_TPEX = set(os.environ['ON_TPEX'].split(','))


def to_date(time: Arrow | int | str, timezone: str = ''):
    if not isinstance(time, Arrow):
        time = arrow.get(time)
    if timezone:
        time = time.to(timezone)
    return time.format('YYYY-MM-DD')


def gen_dates(start: Arrow, end: Arrow):
    return map(to_date, Arrow.range('day', start, end))


def get_suffix(market: Literal['c', 't', 'u'], symbol: str):
    return '.TW' + ('O' if symbol in ON_TPEX else '') if market == 't' else ''


@cached(43200)
def get_text(url: str):
    return r.get(url).text


async def get_today_dividend(sess: AsyncClient, market: Literal['t', 'u'], symbol: str):
    today = to_date(arrow.now(MARKET_TO_TIMEZONE[market]))
    if market == 't':
        text = get_text('https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json')
        for e in json.loads(text)['data']:
            y, m, d = map(int, re.findall('\\d+', e[0]))
            if to_date(Arrow(y + 1911, m, d)) == today and e[1] == symbol:
                return float(e[7])
    else:
        res = await sess.get(
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


def get_sorted_values(D: dict[str, float]):
    return np.array([e[1] for e in sorted(D.items(), key=lambda x: x[0])])


@nb.njit
def break_limit(prices: Array[f8]):
    L = np.abs((prices[1:] - prices[:-1]) / prices[:-1]) > 0.09
    s = 0
    for i, l in enumerate(L):
        if not s and l:
            s = i + 1
        elif s and not l:
            prices[s : i + 1] = prices[i + 1]
            s = 0
    if s:
        prices[s:] = prices[-1]
    return prices


@nb.njit
def post_process(prices: Array[f8], n: int, limited: bool = False):
    prices = np.trim_zeros(prices, 'f')
    assert len(prices) >= n
    for i in range(1, len(prices)):
        if not prices[i]:
            prices[i] = prices[i - 1]
    assert np.all(np.abs(np.log(prices[1:] / prices[:-1])) < np.log(2))
    if limited:
        prices = break_limit(prices)
    return prices[-n:]
