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
    S = np.exp(np.mean(R_, 1) * 182) / np.sqrt(np.mean(D**2, 1)) * slots
    # S = np.exp(np.mean(R_, 1) * 182) / np.mean(D, 1) * slots
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
async def get_prices(market: Literal['c', 't', 'u'], symbol: str, n: int):
    return (
        await (
            crypto.get_prices(symbol, n)
            if market == 'c'
            else stock.get_prices(market, symbol, n, False)
        )
    ).tolist()


@app.get("/leverage")
async def get_leverage(
    market: Literal["c", "t", "u"], symbol: str, max_l: float
) -> float:
    EMA_W = 91
    Z_W = 364

    alpha = 2.0 / (EMA_W + 1)
    k = calc_k(EMA_W)
    n = Z_W + k - 1

    prices = await (
        crypto.get_prices(symbol, n)
        if market == "c"
        else stock.get_prices(market, symbol, n)
    )

    ema = calc_ema(prices, alpha, k)

    dev = np.log(prices[-Z_W:] / ema)
    x = dev[-1] / dev.std()

    a = 1.768825
    offset = (1.0 + max_l) / 2.0
    scale = (1.0 - max_l) / (2.0 * a)
    lev = offset + scale * x
    return float(lev)
