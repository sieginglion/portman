from dataclasses import dataclass
from typing import Literal

import uvicorn
from fastapi import FastAPI
from numpy import float64 as f8

from position import Position, Signal

app = FastAPI()


@dataclass
class Derived:
    expected: f8
    omega: f8
    downside: f8
    signals: tuple[Signal, Signal]


# @app.get('/ranking')
# def get_ranking(k: int) -> HTMLResponse:
#     return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(
    market: Literal['c', 't', 'u'], symbol: str, metric_scale: int, theta: float
) -> Derived:
    p = await Position(market, symbol, metric_scale)
    return Derived(
        **p.calc_metrics(theta), signals=(p.calc_signal(91), p.calc_signal(182))
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
