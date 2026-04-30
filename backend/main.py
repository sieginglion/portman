import asyncio
import io
import logging
import os
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
    # W = (
    #     P
    #     * (np.maximum(R_, 0) ** 1).mean(1) ** 0.5
    #     / (np.maximum(-R_, 0) ** 1).mean(1) ** 1
    # )
    W = P * np.maximum(R_, 0).mean(1) / np.maximum(-R_, 0).mean(1)
    W /= W.sum()
    return W
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
async def get_scores(
    market: Literal['c', 't', 'u'],
    symbol: str,
    q: int,
    end_date: str = '',
    ema7: bool = False,
):
    if market == 'c' and symbol == 'BTC':
        return await calc_btc_score(q), None
    if not end_date:
        end_date = str(pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date())
    return await valuation.calc_scores(market, symbol, end_date, q, ema7)


@app.get('/growths')
async def get_growths(market: Literal['c', 't', 'u'], symbol: str):
    if market == 'c' and symbol == 'BTC':
        return await calc_btc_growth(), None
    xps = await valuation.fetch_xps(market, symbol, 5)
    r, e = xps['rps'], xps['eps']
    return (
        r.iloc[-1] / r.iloc[-5] - 1,
        e.iloc[-1] / e.iloc[-5] - 1 if (e > 0).all() else None,
    )


async def get_btc_prices_and_4y_ema(n: int):
    from .position import calc_ema, calc_k

    k = calc_k(1456)
    prices = await shared.get_prices('c', 'BTC', n + k - 1, False)
    ema = calc_ema(prices, 2 / (1456 + 1), k)
    return prices[-n:], ema


async def calc_btc_growth():
    _, ema = await get_btc_prices_and_4y_ema(365)
    return ema[-1] / ema[0] - 1


async def calc_btc_score(q: int):
    prices, ema = await get_btc_prices_and_4y_ema(91 * q)
    ratio = prices / ema
    return (ratio < ratio[-1]).mean()


@app.get('/pegs')
async def get_pegs(
    market: Literal['t', 'u'],
    symbol: str,
    q: int,
):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    from matplotlib import pyplot as plt

    peg = await valuation.calc_pegs(market, symbol, q)
    if peg.empty:
        raise fastapi.HTTPException(422, 'No positive historical PEG values')

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.scatter(peg.index, peg, s=10, color='#2563eb', alpha=0.72, linewidths=0)
    ax.scatter(peg.index[-1:], peg.iloc[-1:], s=32, color='#dc2626', linewidths=0)
    ax.axhline(peg.iloc[-1], color='#dc2626', linewidth=1, alpha=0.45)
    ax.set_title(f'{symbol} historical PEG, last {q} quarters')
    ax.set_ylabel('PEG')
    ax.grid(True, axis='y', color='#e5e7eb', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return fastapi.Response(buf.getvalue(), media_type='image/png')


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
