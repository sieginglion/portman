from typing import Literal, TypedDict

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from numpy import float64 as f8

from position import Position, Signal, calc_signals

app = FastAPI()
app.mount('/app', StaticFiles(directory='.'))


class Derived(TypedDict):
    expected: f8
    omega: f8
    downside: f8
    signals: tuple[Signal, Signal]


@app.get('/derived')
async def get_derived(
    market: Literal['c', 't', 'u'], symbol: str, metric_scale: int, theta: float
) -> Derived:
    p = await Position(market, symbol, metric_scale)
    return {**p.calc_metrics(theta), 'signals': (p.calc_signal(91), p.calc_signal(182))}


class Chart(TypedDict):
    P: list
    X_b: list
    X_s: list
    Y_b: list
    Y_s: list


@app.get('/chart')
async def get_chart(market: Literal['c', 't', 'u'], symbol: str) -> Chart:
    p = await Position(market, symbol, 364)
    w_s, w_l, _ = p.calc_signal(91)
    S = calc_signals(p.prices, w_s, w_l)[-91:]
    P = p.prices[-91:]
    return {
        'P': P.tolist(),
        'X_b': np.where(S > 0)[0].tolist(),
        'Y_b': P[S > 0].tolist(),
        'X_s': np.where(S < 0)[0].tolist(),
        'Y_s': P[S < 0].tolist(),
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
