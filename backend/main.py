import asyncio
import logging
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
from general_cache import cached
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import stock
from .position import Position, calc_k

app = fastapi.FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/content')
async def get_content(url: str):
    async with AsyncClient(timeout=60) as sess:
        r = await sess.get(unquote(url))
        return fastapi.Response(r.content, r.status_code, r.headers)


def calc_weights(R: Array[f8], t: float = 0) -> Array[f8]:
    R_ = R - t
    U, D = R_ * (R_ > 0), np.abs(R_ * (R_ < 0))
    A = np.mean(U, 1) ** 0.25 / np.mean(D, 1)
    B = np.mean(U**2, 1) ** 0.125 / np.mean(D**2, 1) ** 0.5
    S = (A * B) ** 0.5
    W = S / np.sum(S)
    u = np.log(np.exp(np.sum(R, 1)) @ W) / R.shape[1]
    if abs(u) < 1e-4 or abs(t - u) < 1e-6:
        return W
    return calc_weights(R, u)


@app.post('/weights')
# @cached(43200)
async def get_weights(positions: list[tuple[Literal['c', 't', 'u'], str]]):
    P = np.array(
        [
            p.prices
            for p in await asyncio.gather(*[Position(m, s, 365) for m, s in positions])
        ]
    )
    return calc_weights(np.log(P[:, 1:] / P[:, :-1])).tolist()


@app.get('/charts')
@cached(600)
async def get_charts(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 364 + calc_k(182))
    S, L = p.calc_signals(91), p.calc_signals(182)
    P = p.prices.tolist()
    return (
        (P[-91:], np.where(S > 0)[0].tolist(), np.where(S < 0)[0].tolist()),
        (P[-182:], np.where(L > 0)[0].tolist(), np.where(L < 0)[0].tolist()),
    )


@app.get('/prices')
async def get_prices(symbol: str, n: int):
    market = 't' if symbol[0].isnumeric() else 'u'
    return (await stock.get_prices(market, symbol, n, False)).tolist()
