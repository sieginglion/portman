import asyncio
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import arrow
import dotenv
import numba as nb
import numpy as np
import pandas as pd
import requests as r
import uvicorn
from arrow import Arrow
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from httpx import AsyncClient
from more_itertools import islice_extended as islice
from more_itertools import last
from numpy import float64 as f8
from numpy.typing import NDArray as Array

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

FMP_KEY = os.environ['FMP_KEY']
INF = float('inf')
MARKET_TO_TIMEZONE = {'c': 'UTC', 't': 'Asia/Taipei', 'u': 'America/New_York'}


def to_date(time: Any, timezone: str = '') -> str:
    if isinstance(time, Arrow):
        if timezone:
            time.to(timezone)
        else:
            pass
    else:
        if timezone:
            time = arrow.get(time, tzinfo=timezone)
        else:
            time = arrow.get(time)
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
    return round(math.log(0.05) / math.log(1 - 2 / (w + 1)))


@nb.njit(parallel=True)
def calc_ema(A: Array[f8], a: float, k: int) -> Array[f8]:
    K = a * (1 - a) ** np.arange(k - 1, -1, -1)
    K = K / np.sum(K)
    return np.array([np.sum(A[i : i + k] * K) for i in range(len(A) - k + 1)])
