import logging
from typing import Literal

from fastapi import FastAPI

from .position import Position, calc_k

app = FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get('/score-and-signals')
async def get_score_and_signals(
    market: Literal['c', 't', 'u'], symbol: str, score_scale: int
):
    p = await Position(market, symbol, max(score_scale + 1, 364 + calc_k(182)))
    return {
        'score': p.calc_score(score_scale),
        'signals': {91: p.calc_signal(91), 182: p.calc_signal(182)},
    }
