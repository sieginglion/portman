import logging
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
from httpx import AsyncClient

from .position import Position, calc_k

app = fastapi.FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/content')
async def get_content(url: str):
    async with AsyncClient(timeout=60) as sess:
        r = await sess.get(unquote(url))
        return fastapi.Response(r.content, r.status_code, r.headers)


@app.get('/score')
async def get_score(market: Literal['c', 't', 'u'], symbol: str, scale: int):
    p = await Position(market, symbol, scale + 1)
    return p.calc_score(scale)


@app.get('/signals')
async def get_signals(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 363 + calc_k(182))
    S, L = p.calc_signals(91), p.calc_signals(182)
    return (int(S[-1]), int(L[-1]))


@app.get('/charts')
async def get_charts(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 363 + calc_k(182))
    P = p.prices.tolist()
    S, L = p.calc_signals(91), p.calc_signals(182)
    return (
        (P[-91:], np.where(S > 0)[0].tolist(), np.where(S < 0)[0].tolist()),
        (P[-182:], np.where(L > 0)[0].tolist(), np.where(L < 0)[0].tolist()),
    )
