import asyncio
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, TypedDict

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
from numpy import float64 as f8
from numpy.typing import NDArray as Array

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

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


def get_values(D: dict[str, float]) -> Array[f8]:
    return np.array([e[1] for e in sorted(D.items(), key=lambda x: x[0])])


@nb.njit
def get_patched(A: Array[f8]) -> Array[f8]:
    for i in range(1, len(A)):
        if not A[i] and A[i - 1]:
            A[i] = A[i - 1]
    return A


@nb.njit(parallel=True)
def calc_ema(A: Array[f8], a: float, k: int) -> Array[f8]:
    K = a * (1 - a) ** np.arange(k - 1, -1, -1)
    K = K / np.sum(K)
    return np.array([np.sum(A[i : i + k] * K) for i in range(len(A) - k + 1)])
