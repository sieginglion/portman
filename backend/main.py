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

from . import crypto, stock
from .position import Position, calc_ema, calc_k

app = fastapi.FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/content')
async def get_content(url: str):
    async with AsyncClient(timeout=60) as sess:
        r = await sess.get(unquote(url))
        return fastapi.Response(r.content, r.status_code, r.headers)


def calc_weights(R: Array[f8], slots: list[float], t: float = 0) -> Array[f8]:
    R_ = R - t
    U, D = np.maximum(R_, 0), np.maximum(-R_, 0)
    # S = np.exp(np.mean(R_, 1) * 182) / np.sqrt(np.mean(D ** 2, 1)) * slots
    S = np.exp(np.mean(R_, 1) * 182) / np.mean(D, 1) * slots
    W = S / np.sum(S)
    u = np.log(np.exp(np.sum(R, 1)) @ W) / R.shape[1]
    if abs(u) < 1e-4 or abs(t - u) < 1e-6:
        return W
    return calc_weights(R, slots, u)


@app.post('/weights')
async def get_weights(
    positions: list[tuple[Literal['c', 't', 'u'], str]], slots: list[float]
):
    P = np.array(
        [
            p.prices
            for p in await asyncio.gather(*[Position(m, s, 365) for m, s in positions])
        ]
    )
    return calc_weights(np.log(P[:, 1:] / P[:, :-1]), slots).tolist()


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
    if symbol.endswith('.c'):
        market, symbol = 'c', symbol[:-2]
    elif symbol[0].isdigit():
        market = 't'
    else:
        market = 'u'
    return (await crypto.get_prices(symbol, n) if market == 'c' else stock.get_prices(market, symbol, n, False)).tolist()


@app.post('/leverage')
async def get_leverage(
    positions: list[tuple[Literal['c', 't', 'u'], str]], weights: list[float]
):
    scale = 91
    w = 91
    n = scale + calc_k(w) - 1
    P = np.array(
        await asyncio.gather(
            *[
                (crypto.get_prices(p[1], n) if p[0] == 'c' else stock.get_prices(*p, n))
                for p in positions
            ]
        )
    )
    R = weights @ (P / P[:, [0]])
    B = calc_ema(R, 2 / (w + 1), calc_k(w))
    R = R[-scale:]
    L = B / R
    return L[-1] / np.median(L)
