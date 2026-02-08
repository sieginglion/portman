import asyncio
import logging
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import crypto, shared, stock, valuation

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
    return W if abs(u) < 3e-5 or abs(u - t) < 3e-7 else calc_weights(P, R, u)


# TODO: numpy array typing with shape
@app.post('/weights')
async def get_weights(
    positions: list[tuple[Literal['c', 't', 'u'], str]], prior: list[float]
):
    P = np.array(
        await asyncio.gather(
            *[shared.get_prices(m, s, 365, True) for m, s in positions]
        )
    )
    return calc_weights(np.array(prior), np.log(P[:, 1:] / P[:, :-1])).tolist()


@app.get('/px-score')
async def get_px_score(market: Literal['t', 'u'], symbol: str):
    return await valuation.calc_px_score(market, symbol)


@app.get('/prices')
async def get_prices(market: Literal['c', 't', 'u'], symbol: str, n: int):
    return (
        await (
            crypto.get_prices(symbol, n)
            if market == 'c'
            else stock.get_prices(market, symbol, n, False)
        )
    ).tolist()
