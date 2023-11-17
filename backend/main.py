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

from .position import Position, calc_k

app = fastapi.FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/content')
async def get_content(url: str):
    async with AsyncClient(timeout=60) as sess:
        r = await sess.get(unquote(url))
        return fastapi.Response(r.content, r.status_code, r.headers)


def calc_weights(R: Array[f8], V: Array[f8], t: float) -> Array[f8]:
    # R_ = (1 + R) / (1 + t) - 1
    R_ = R - t
    S = 1 / (R_ * (R_ < 0) @ V)
    W = S / np.sum(S)
    u = (R @ V @ W).item()
    if np.abs(u - t) < 1e-6:
        return W
    return calc_weights(R, V, u)


@app.post('/weights')
@cached(43200)
async def get_weights(positions: tuple[tuple[Literal['c', 't', 'u'], str], ...]):
    P = np.array(
        [
            p.prices
            for p in await asyncio.gather(*[Position(m, s, 410) for m, s in positions])
        ]
    )
    R = P[:, 1:] / P[:, :-1] - 1
    V = np.append(np.arange(1, 91) / (91 * 364), np.full(319, 1 / 364))
    return calc_weights(R, V, 0).tolist()


@app.get('/signals')
@cached(600)
async def get_signals(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 364 + calc_k(182))
    S, L = p.calc_signals(91), p.calc_signals(182)
    return (int(S[-1]), int(L[-1]))


@app.get('/charts')
async def get_charts(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 364 + calc_k(182))
    S, L = p.calc_signals(91), p.calc_signals(182)
    P = p.prices.tolist()
    return (
        (P[-91:], np.where(S > 0)[0].tolist(), np.where(S < 0)[0].tolist()),
        (P[-182:], np.where(L > 0)[0].tolist(), np.where(L < 0)[0].tolist()),
    )
