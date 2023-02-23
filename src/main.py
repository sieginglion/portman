from dataclasses import dataclass
from typing import Literal

import uvicorn
from fastapi import FastAPI
from numpy import float64 as f8

import position

app = FastAPI()


@dataclass
class Derived:
    ER: f8
    omega: f8
    downside: f8
    w_s: int
    w_l: int
    signal: f8


# @app.get('/ranking')
# def get_ranking(k: int) -> HTMLResponse:
#     return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(
    market: Literal['c', 't', 'u'], symbol: str, scale: int, theta: float
) -> Derived:
    p = await position.Position(market, symbol, scale)
    return Derived(**p.calc_metrics(theta), **p.calc_signal())


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
