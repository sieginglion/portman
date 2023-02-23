import logging
from typing import Literal, TypedDict

import numba as nb
import numpy as np
from numpy import float64 as f8
from numpy.typing import NDArray as Array

import crypto
import stock
from shared import INF, calc_ema, calc_k


def calc_signals(prices: Array[f8], w_s: int, w_l: int) -> Array[f8]:
    s_ema = calc_ema(prices, 2 / (w_s + 1), calc_k(w_s))
    l_ema = calc_ema(prices, 2 / (w_l + 1), calc_k(w_l))
    macd = s_ema[-len(l_ema) :] - l_ema
    slow = calc_ema(macd, 2 / (2 + 1), calc_k(2))
    macd = macd[-len(slow) :]
    signals = np.zeros(len(slow))
    signals[(macd < 0) & (macd > slow)] = 1
    signals[(macd > 0) & (macd < slow)] = -1
    return signals


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8], min_trades: int) -> float:
    cash = position = value = 0
    last_side = trades = 0
    for i, price in enumerate(prices):
        if i and price != prices[i - 1]:
            value = position * price
            if value < 1000 and signals[i] > 0:
                cash -= 1000 - value
                position = 1000 / price
                last_side = 1
            elif value > 1000 and signals[i] < 0:
                cash += value - 1000
                position = 1000 / price
                trades += 1 if last_side == 1 else 0
                last_side = -1
    return cash + value if trades >= min_trades else -INF


class Metrics(TypedDict):
    ER: f8
    omega: f8
    downside: f8


class Signal(TypedDict):
    w_s: int
    w_l: int
    signal: f8


class Position:
    def __init__(self, market: Literal['c', 't', 'u'], symbol: str, scale: int) -> None:
        self.market = market
        self.symbol = symbol
        self.scale = scale

    async def ainit(self):
        n = self.scale + 1
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)  # type: ignore
        )
        if len(self.prices) != n:
            raise ValueError
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_metrics(self, theta: float) -> Metrics:
        R = np.log(self.prices[1:] / self.prices[:-1]) - theta
        return {
            'ER': np.mean(R) + theta,
            'omega': np.sum(R[R > 0]) / -np.sum(R[R < 0]),
            'downside': (np.sum(R[R < 0] ** 2) / len(R)) ** 0.5,
        }

    def calc_signal(self, min_trades: int) -> Signal:
        s = 183
        W_to_score = {
            (w_s, w_l): simulate(
                self.prices[-s:], calc_signals(self.prices, w_s, w_l)[-s:], min_trades
            )
            for w_s in range(7, 92, 7)
            for w_l in range(7, 92, 7)
            if w_s < w_l
        }
        (w_s, w_l), score = max(W_to_score.items(), key=lambda x: x[1])
        logging.info((w_s, w_l, score))
        return {
            'w_s': w_s,
            'w_l': w_l,
            'signal': calc_signals(self.prices, w_s, w_l)[-1],
        }


# import asyncio


# async def main():
#     p = await Position('c', 'ETH', 364 * 2)
#     print(p.calc_metrics(0))
#     print(p.calc_signal(1))


# asyncio.run(main())
