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
# logging.basicConfig(level=logging.INFO)

PERCENTILE_BANDS = np.linspace(0, 100, 9)
PERCENTILE_BAND_COLORS = [
    '#FF0000',
    '#FFBF00',
    '#80FF00',
    '#00FF40',
    '#00FFFF',
    '#0040FF',
    '#8000FF',
    '#FF00BF',
]


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
    positions: list[tuple[Literal['c', 'j', 't', 'u'], str]], prior: list[float]
):
    P = np.array(
        await asyncio.gather(
            *[shared.get_prices(m, s, 365, True) for m, s in positions]
        )
    )
    return calc_weights(np.array(prior), np.log(P[:, 1:] / P[:, :-1])).tolist()


def add_percentile_bands(ax, series: pd.Series):
    values = series.dropna()
    if values.empty:
        return

    levels = np.percentile(values, PERCENTILE_BANDS)
    for lo, hi, color in zip(levels, levels[1:], PERCENTILE_BAND_COLORS):
        ax.axhspan(lo, hi, color=color, alpha=0.4, zorder=0)

    for level in levels:
        ax.axhline(level, color='#111827', linewidth=0.8, alpha=0.35, zorder=1)


@app.get('/scores')
async def get_scores(
    market: Literal['c', 'j', 't', 'u'],
    symbol: str,
    q: int,
):
    if market == 'c' and symbol == 'BTC':
        return await calc_btc_score(q), None

    df = await valuation.calc_px(market, symbol, q)

    def score(s: pd.Series) -> float:
        return (s < s.iloc[-1]).mean()

    def pe_score(s: pd.Series) -> float | None:
        if s.isna().any():
            return None
        return score(s)

    return score(df['ps']), pe_score(df['pe'])


@app.get('/growths')
async def get_growths(market: Literal['c', 'j', 't', 'u'], symbol: str):
    if market == 'c' and symbol == 'BTC':
        return await calc_btc_growth()
    xps = await valuation.fetch_xps(market, symbol, 5, include_eps=False)
    r = xps['rps']
    return r.iloc[-1] / r.iloc[-5] - 1


async def get_btc_prices_and_4y_sma(n: int):
    window = 1456
    prices = await shared.get_prices('c', 'BTC', n + window - 1, False)
    cumsum = np.cumsum(prices, dtype=np.float64)
    cumsum = np.concatenate(([0.0], cumsum))
    sma = (cumsum[window:] - cumsum[:-window]) / window
    # from .position import calc_ema, calc_k
    # k = calc_k(1456)
    # prices = await shared.get_prices('c', 'BTC', n + k - 1, False)
    # return prices[-n:], calc_ema(prices, 2 / (1456 + 1), k)
    return prices[-n:], sma


async def calc_btc_growth():
    _, sma = await get_btc_prices_and_4y_sma(365)
    return sma[-1] / sma[0] - 1


async def calc_btc_score(q: int):
    prices, sma = await get_btc_prices_and_4y_sma(91 * q)
    ratio = prices / sma
    return (ratio < ratio[-1]).mean()


async def calc_btc_ps(q: int) -> pd.Series:
    prices, sma = await get_btc_prices_and_4y_sma(91 * q)
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE['c']).date(),
        periods=len(prices),
    )
    return pd.Series(prices / sma, index=index)


@app.get('/pegs')
async def get_pegs(
    market: Literal['j', 't', 'u'],
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


@app.get('/px')
async def get_px(
    market: Literal['c', 'j', 't', 'u'],
    symbol: str,
    q: int,
):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    from matplotlib import pyplot as plt

    if market == 'c':
        if symbol != 'BTC':
            raise fastapi.HTTPException(422, 'Only BTC is supported for crypto /px')

        ps = await calc_btc_ps(q)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=160)
        add_percentile_bands(ax, ps)
        ax.plot(ps.index, ps, color='#f97316', linewidth=1.5, zorder=2)
        ax.scatter(
            ps.index[-1:],
            ps.iloc[-1:],
            s=28,
            color='#f97316',
            linewidths=0,
            zorder=3,
        )
        ax.set_title(f'BTC historical P/S, last {q} quarters')
        ax.set_ylabel('P/S')
        ax.spines[['top', 'right']].set_visible(False)

        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        return fastapi.Response(buf.getvalue(), media_type='image/png')

    px = await valuation.calc_px(market, symbol, q)
    ps = px['ps']
    pe = px['pe'].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=160, sharex=True)
    ps_ax, pe_ax = axes

    add_percentile_bands(ps_ax, ps)
    ps_ax.plot(ps.index, ps, color='#2563eb', linewidth=1.5, zorder=2)
    ps_ax.scatter(
        ps.index[-1:], ps.iloc[-1:], s=28, color='#2563eb', linewidths=0, zorder=3
    )
    ps_ax.set_title(f'{symbol} historical P/S, last {q} quarters')
    ps_ax.set_ylabel('P/S')
    ps_ax.spines[['top', 'right']].set_visible(False)

    if not pe.empty:
        add_percentile_bands(pe_ax, pe)
        pe_ax.plot(pe.index, pe, color='#dc2626', linewidth=1.5, zorder=2)
        pe_ax.scatter(
            pe.index[-1:], pe.iloc[-1:], s=28, color='#dc2626', linewidths=0, zorder=3
        )

    pe_ax.set_title(f'{symbol} historical P/E, last {q} quarters')
    pe_ax.set_ylabel('P/E')
    pe_ax.spines[['top', 'right']].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return fastapi.Response(buf.getvalue(), media_type='image/png')


@app.get('/prices')
async def get_prices(
    market: Literal['c', 'j', 't', 'u'], symbol: str, n: int, ema7: bool = False
):
    return (await shared.get_prices(market, symbol, n, False, ema7)).tolist()


@app.get('/extremum')
async def get_extremum(
    market: Literal['c', 'j', 't', 'u'], symbol: str, n: int, max: bool = False
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
