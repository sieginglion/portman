import asyncio
import io
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import fastapi
import numpy as np
import pandas as pd
from httpx import AsyncClient
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import shared, valuation
from .call_recorder import ValuationCallRecorder

CALL_RECORDER_ENABLED = (
    os.getenv("ENABLE_VALUATION_CALL_RECORDING", "").lower() == "true"
)


@asynccontextmanager
async def lifespan(_: fastapi.FastAPI) -> AsyncGenerator[None, None]:
    recorder = None
    if CALL_RECORDER_ENABLED:
        recorder = ValuationCallRecorder(valuation)
        recorder.enable()

    try:
        yield
    finally:
        if recorder is not None:
            try:
                recorder.write(Path.cwd() / "portman-valuation-call-edges.json")
            finally:
                recorder.disable()


app = fastapi.FastAPI(lifespan=lifespan)
# logging.basicConfig(level=logging.INFO)

PERCENTILE_BANDS = np.linspace(0, 100, 9)
PERCENTILE_BAND_COLORS = [
    "#FF0000",
    "#FFBF00",
    "#80FF00",
    "#00FF40",
    "#00FFFF",
    "#0040FF",
    "#8000FF",
    "#FF00BF",
]
BTC_QUARTER_DAYS = 91
BTC_GROWTH_DAYS = 365
BTC_GROWTH_SMA_WINDOW_DAYS = BTC_QUARTER_DAYS * 16


def is_btc(market: str, symbol: str) -> bool:
    return market == "c" and symbol == "BTC"


@app.get("/content")
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
@app.post("/weights")
async def get_weights(
    positions: list[tuple[Literal["c", "j", "t", "u"], str]], prior: list[float]
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
    for lo, hi, color in zip(levels, levels[1:], PERCENTILE_BAND_COLORS, strict=False):
        ax.axhspan(lo, hi, color=color, alpha=0.4, zorder=0)

    for level in levels:
        ax.axhline(level, color="#111827", linewidth=0.8, alpha=0.35, zorder=1)


def add_quarter_end_lines(ax, series: pd.Series):
    quarter_ends = pd.date_range(series.index.min(), series.index.max(), freq="QE")
    for quarter_end in quarter_ends:
        ax.axvline(
            quarter_end,
            color="#111827",
            linewidth=0.8,
            linestyle=":",
            alpha=0.35,
            zorder=1,
        )


def plot_valuation_series(
    ax,
    series: pd.Series,
    symbol: str,
    multiple: str,
    q: int,
    color: str,
):
    if not series.empty:
        add_percentile_bands(ax, series)
        add_quarter_end_lines(ax, series)
        ax.plot(series.index, series, color=color, linewidth=1.5, zorder=2)
        ax.scatter(
            series.index[-1:],
            series.iloc[-1:],
            s=28,
            color=color,
            linewidths=0,
            zorder=3,
        )

    ax.set_title(f"{symbol} historical {multiple}, last {q} quarters")
    ax.set_ylabel(multiple)
    ax.spines[["top", "right"]].set_visible(False)


def figure_to_png_response(fig, plt) -> fastapi.Response:
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png")
        return fastapi.Response(buf.getvalue(), media_type="image/png")
    finally:
        plt.close(fig)


def calc_downside_score(s: pd.Series, valuation_score_upper: float) -> float:
    if s.empty or s.isna().any() or (s <= 0).any():
        raise ValueError

    upper = float(s.quantile(valuation_score_upper))
    lower = float(s.quantile(valuation_score_upper / 2))
    if lower == upper:
        return 0

    return float(max(upper - s.iloc[-1], 0) / (upper - lower))


@app.get("/scores")
async def get_scores(
    market: Literal["c", "j", "t", "u"],
    symbol: str,
    q: int,
    u: float,
):
    if is_btc(market, symbol):
        return await calc_btc_score(q, u), None

    df = await valuation.calc_px(market, symbol, q)
    ps, pe = df["ps"], df["pe"]
    return (
        None if pd.isna(ps.iloc[0]) else calc_downside_score(ps, u),
        None if pd.isna(pe.iloc[0]) else calc_downside_score(pe, u),
    )


@app.get("/diagnostics")
async def get_diagnostics():
    return valuation.get_xps_diagnostics()


@app.get("/growth")
async def get_growth(market: Literal["c", "j", "t", "u"], symbol: str):
    if is_btc(market, symbol):
        return await calc_btc_growth()

    xps = await valuation.fetch_xps(market, symbol, 5, include_eps=False)
    r = xps["rps"]
    return r.iloc[-1] / r.iloc[-5] - 1


async def get_btc_prices_and_sma(n: int, window: int):
    prices = await shared.get_prices("c", "BTC", n + window - 1, False)
    cumsum = np.cumsum(prices, dtype=np.float64)
    cumsum = np.concatenate(([0.0], cumsum))
    sma = (cumsum[window:] - cumsum[:-window]) / window
    return prices[-n:], sma


async def get_btc_price_sma_frame(periods: int, window: int) -> pd.DataFrame:
    prices, sma = await get_btc_prices_and_sma(periods, window)
    index = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE["c"]).date(),
        periods=periods,
    )
    return pd.DataFrame({"price": prices, "sma": sma}, index=index)


async def get_btc_price_to_sma_series(periods: int, window: int) -> pd.Series:
    df = await get_btc_price_sma_frame(periods, window)
    return df["price"] / df["sma"]


async def calc_btc_growth():
    df = await get_btc_price_sma_frame(BTC_GROWTH_DAYS, BTC_GROWTH_SMA_WINDOW_DAYS)
    sma = df["sma"]
    return sma.iloc[-1] / sma.iloc[0] - 1


async def get_btc_valuation_proxy(q: int) -> pd.Series:
    periods = BTC_QUARTER_DAYS * q
    window = BTC_QUARTER_DAYS * 16
    return await get_btc_price_to_sma_series(periods, window)


async def calc_btc_score(q: int, valuation_score_upper: float) -> float:
    return calc_downside_score(await get_btc_valuation_proxy(q), valuation_score_upper)


@app.get("/px")
async def get_px(
    market: Literal["c", "j", "t", "u"],
    symbol: str,
    q: int,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from matplotlib import pyplot as plt

    if is_btc(market, symbol):
        ps = await get_btc_valuation_proxy(q)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=160)
        plot_valuation_series(ax, ps, "BTC", "P/S", q, "#f97316")

        return figure_to_png_response(fig, plt)

    px = await valuation.calc_px(market, symbol, q)
    ps = px["ps"].dropna()
    pe = px["pe"].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=160, sharex=True)
    ps_ax, pe_ax = axes

    plot_valuation_series(ps_ax, ps, symbol, "P/S", q, "#2563eb")
    plot_valuation_series(pe_ax, pe, symbol, "P/E", q, "#dc2626")

    return figure_to_png_response(fig, plt)


@app.get("/prices")
async def get_prices(market: Literal["c", "j", "t", "u"], symbol: str, n: int):
    return (await shared.get_prices(market, symbol, n, False)).tolist()


@app.get("/extremum")
async def get_extremum(
    market: Literal["c", "j", "t", "u"], symbol: str, n: int, max: bool = False
):
    prices = await shared.get_prices(market, symbol, n, True)
    i = prices.argmax() if max else prices.argmin()
    dates = pd.date_range(
        end=pd.Timestamp.now(shared.MARKET_TO_TIMEZONE[market]).date(),
        periods=len(prices),
    )
    return dates[i].strftime("%Y-%m-%d"), prices[i]


@app.get("/BTCXAU")
async def get_btcxau():
    btc, paxg = await asyncio.gather(
        shared.get_prices("c", "BTC", 1456, False),
        shared.get_prices("c", "PAXG", 1456, False),
    )
    ratio = btc / paxg
    lo, hi = ratio.min(), ratio.max()
    return float((ratio[-1] - lo) / (hi - lo))


@app.get("/shares")
async def get_shares(symbol: str):
    async with AsyncClient() as sess:
        res = await sess.get(
            "https://financialmodelingprep.com/stable/income-statement",
            params={
                "apikey": shared.FMP_KEY,
                "limit": 1,
                "period": "quarter",
                "symbol": symbol,
            },
        )
    return res.json()[0]["weightedAverageShsOutDil"]
