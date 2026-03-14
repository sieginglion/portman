import asyncio
import logging
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
import pandas as pd
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import shared, valuation

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
        * (np.maximum(R_, 0) ** 1).mean(1) ** 0.5
        / (np.maximum(-R_, 0) ** 1).mean(1) ** 1
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


@app.get('/scores')
async def get_scores(market: Literal['t', 'u'], symbol: str, end_date: str, q: int):
    return await valuation.calc_scores(market, symbol, end_date, q)


@app.get('/prices')
async def get_prices(
    market: Literal['c', 't', 'u'], symbol: str, n: int, ema7: bool = False
):
    return (await shared.get_prices(market, symbol, n, False, ema7)).tolist()


@app.get('/extremum')
async def get_extremum(
    market: Literal['c', 't', 'u'], symbol: str, n: int, max: bool = False
):
    ema = await shared.get_prices(market, symbol, n, True, True)
    i = ema.argmax() if max else ema.argmin()
    dates = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(ema),
    )
    return dates[i].strftime('%Y-%m-%d'), ema[i]


@app.get('/BTCXAU')
async def get_btcxau():
    btc, paxg = await asyncio.gather(
        shared.get_prices('c', 'BTC', 1456, False),
        shared.get_prices('c', 'PAXG', 1456, False),
    )
    ratio = btc / paxg
    lo, hi = ratio.min(), ratio.max()
    return float((ratio[-1] - lo) / (hi - lo))


@app.get('/shares')
async def get_shares(symbol: str):
    async with AsyncClient() as sess:
        res = await sess.get(
            'https://financialmodelingprep.com/stable/income-statement',
            params={
                'apikey': shared.FMP_KEY,
                'limit': 1,
                'period': 'quarter',
                'symbol': symbol,
            },
        )
    return res.json()[0]['weightedAverageShsOutDil']
