import logging
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
from httpx import AsyncClient

from .position import Position, calc_k

app = fastapi.FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/proxy')
async def proxy(url: str):
    async with AsyncClient(timeout=60) as h:
        res = await h.get(unquote(url))
        return fastapi.Response(res.content, res.status_code, res.headers)


@app.get('/score')
async def get_score(market: Literal['c', 't', 'u'], symbol: str, scale: int):
    p = await Position(market, symbol, scale + 1)
    return p.calc_score(scale)


@app.get('/signals')
async def get_signals(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 364 + calc_k(182))
    s, l = p.calc_signals(91), p.calc_signals(182)
    return ((s.w_s, s.w_l, int(s.values[-1])), (l.w_s, l.w_l, int(l.values[-1])))


@app.get('/charts')
async def get_charts(market: Literal['c', 't', 'u'], symbol: str):
    p = await Position(market, symbol, 364 + calc_k(182))
    P = p.prices.tolist()
    S, L = p.calc_signals(91).values, p.calc_signals(182).values
    return (
        (P[-91:], np.where(S > 0)[0].tolist(), np.where(S < 0)[0].tolist()),
        (P[-182:], np.where(L > 0)[0].tolist(), np.where(L < 0)[0].tolist()),
    )
