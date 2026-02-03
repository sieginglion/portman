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


def calc_weights(P: Array[f8], R: Array[f8], t: float = 0):
    R_ = R - t
    W = (
        P
        * (np.maximum(R_, 0) ** 2).mean(1) ** 0.25
        / (np.maximum(-R_, 0) ** 2).mean(1) ** 0.5
    )
    W /= W.sum()
    u = np.log(np.exp(R.sum(1)) @ W) / R.shape[1]
    return W if abs(u) < 3e-5 or abs(u - t) < 3e-6 else calc_weights(P, R, u)


# TODO: numpy array typing with shape
@app.post('/weights')
async def get_weights(
    positions: list[tuple[Literal['c', 't', 'u'], str]], prior: list[float]
):
    P = np.array(
        [
            p.prices
            for p in await asyncio.gather(*[Position(m, s, 365) for m, s in positions])
        ]
    )
    return calc_weights(np.array(prior), np.log(P[:, 1:] / P[:, :-1])).tolist()


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
async def get_prices(market: Literal['c', 't', 'u'], symbol: str, n: int):
    return (
        await (
            crypto.get_prices(symbol, n)
            if market == 'c'
            else stock.get_prices(market, symbol, n, False)
        )
    ).tolist()


# @app.get('/leverage')
# async def get_leverage(
#     max_l: float, market: Literal['c', 't', 'u'], symbol: str
# ) -> float:
#     EMA_W = 91
#     Z_W = 364

#     alpha = 2.0 / (EMA_W + 1)
#     k = calc_k(EMA_W)
#     n = Z_W + k - 1

#     prices = await (
#         crypto.get_prices(symbol, n)
#         if market == 'c'
#         else stock.get_prices(market, symbol, n)
#     )

#     ema = calc_ema(prices, alpha, k)

#     dev = np.log(prices[-Z_W:] / ema)
#     x = dev[-1] / dev.std()

#     a = 1.768825
#     offset = (1.0 + max_l) / 2.0
#     scale = (1.0 - max_l) / (2.0 * a)
#     lev = offset + scale * x
#     return float(lev)


@app.post('/leverage')
async def get_leverage(
    positions: list[tuple[Literal['c', 't', 'u'], str]],
    weights: list[float],
    max_l: float,
):
    n, w = 364, 91
    k = calc_k(w)
    P = [
        p.prices
        for p in await asyncio.gather(
            *[Position(m, s, n + k - 1) for m, s in positions]
        )
    ]
    P = np.array(weights) @ P / P[:, [0]]
    E = calc_ema(P, 2 / (w + 1), k)
    D = np.log(P[-n:] / E)
    z = D[-1] / D.std()
    a = 1.768825
    offset = (1.0 + max_l) / 2.0
    scale = (1.0 - max_l) / (2.0 * a)
    lev = offset + scale * z
    return float(lev)
