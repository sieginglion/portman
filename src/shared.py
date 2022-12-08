import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, TypedDict

import arrow
import dotenv
import numba as nb
import numpy as np
import requests as r
import uvicorn
from arrow import Arrow
from fastapi import FastAPI
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as A

dotenv.load_dotenv()

FMP_KEY = os.environ['FMP_KEY']
market_to_timezone = {'t': 'Asia/Taipei', 'u': 'America/New_York', 'c': 'UTC'}


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


def get_values(D: dict[str, float]) -> A[f8]:
    return np.array([e[1] for e in sorted(D.items(), key=lambda x: x[0])])


@nb.njit
def get_patched(A: A[f8]) -> A[f8]:
    for i in range(1, len(A)):
        if not A[i] and A[i - 1]:
            A[i] = A[i - 1]
    return A


@nb.njit(parallel=True)
def calc_ema(A: A[f8], a: float, k: int) -> A[f8]:
    K = a * (1 - a) ** np.arange(k - 1, -1, -1)
    K = K / np.sum(K)
    return np.array([np.sum(A[i : i + k] * K) for i in range(len(A) - k + 1)])
