import logging
from typing import Literal
from urllib.parse import unquote

from fastapi import FastAPI, Response
from httpx import AsyncClient

from .position import Position, calc_k

app = FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/proxy')
async def proxy(url: str):
    async with AsyncClient(timeout=60) as h:
        res = await h.get(unquote(url))
        return Response(res.content, res.status_code, res.headers)


@app.get('/score-and-signals')
async def get_score_and_signals(
    market: Literal['c', 't', 'u'], symbol: str, score_scale: int
):
    p = await Position(market, symbol, max(score_scale + 1, 364 + calc_k(182)))
    return {
        'score': p.calc_score(score_scale),
        'signals': {91: p.calc_signal(91), 182: p.calc_signal(182)},
    }
