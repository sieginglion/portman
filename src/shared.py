import json
import logging
from dataclasses import dataclass
from math import ceil, log
from os import environ

import arrow
import numba as nb
import numpy as np
import requests as r
from arrow import Arrow
from dotenv import load_dotenv
from numpy import float64 as f8
from numpy.typing import NDArray as Array

load_dotenv()
logging.basicConfig(level=logging.INFO)

cache = {}
FMP_KEY = environ['FMP_KEY']
INF = float('inf')
MARKET_TO_TIMEZONE = {'c': 'UTC', 't': 'Asia/Taipei', 'u': 'America/New_York'}


@dataclass
class CachedResponse:
    text: str

    def json(self) -> dict:
        return json.loads(self.text)


def cached_get(url: str) -> CachedResponse:
    now = arrow.now().timestamp()
    if (not (cached := cache.get(url))) or (cached['timestamp'] < now - 3600):
        cache[url] = {'text': r.get(url).text, 'timestamp': now}
    return CachedResponse(cache[url]['text'])


def to_date(time: Arrow | float | int | str, timezone: str = '') -> str:
    if not isinstance(time, Arrow):
        time = arrow.get(time)

    if timezone:
        time = time.to(timezone)

    return time.format('YYYY-MM-DD')


def gen_dates(start: Arrow, end: Arrow) -> list[str]:
    return [to_date(a) for a in Arrow.range('day', start, end)]


def get_values(D: dict[str, float]) -> Array[f8]:
    return np.array([e[1] for e in sorted(D.items(), key=lambda x: x[0])])


@nb.njit
def clean_up(A: Array[f8]) -> Array[f8]:
    def diff(a: f8, b: f8) -> f8:
        return np.abs(np.log(a / b))

    for i in range(1, len(A)):
        if A[i - 1] and not A[i]:
            A[i] = A[i - 1]
    for i in range(1, len(A) - 1):
        if A[i - 1] and diff(A[i - 1], A[i]) > 0.693 and diff(A[i], A[i + 1]) > 0.693:
            A[i] = A[i - 1]
    return A


def calc_k(w: int) -> int:
    return ceil(log(0.05) / log(1 - 2 / (w + 1)))


@nb.njit(parallel=True)
def calc_ema(A: Array[f8], a: float, k: int) -> Array[f8]:
    K = a * (1 - a) ** np.arange(k - 1, -1, -1)
    K = K / np.sum(K)
    ema = np.empty(len(A) - k + 1)
    for i in nb.prange(len(A) - k + 1):
        ema[i] = np.sum(A[i : i + k] * K)
    return ema
