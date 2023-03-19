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
    return ((macd < 0) & (macd > slow)).astype(f8) - (
        (macd > 0) & (macd < slow)
    ).astype(f8)


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8]) -> float:
    cash = 1000
    position = 0
    for i, (price, signal) in enumerate(zip(prices[1:], signals[1:]), 1):
        if price != prices[i - 1] and abs(price / prices[i - 1] - 1) < 0.09:
            if signal == 1 and position == 0:
                position = cash / price
                cash = 0
            elif signal == -1 and position > 0:
                cash = position * price
                position = 0
    return cash + position * prices[-1]


class Metrics(TypedDict):
    expected: f8
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
            'expected': np.mean(R) + theta,
            'omega': np.sum(R[R > 0]) / -np.sum(R[R < 0]),
            'downside': (np.sum(R[R < 0] ** 2) / len(R)) ** 0.5,
        }

    def calc_signal(self) -> Signal:
        s = 92
        W_to_score = {
            (w_s, w_l): simulate(
                self.prices[-s:], calc_signals(self.prices, w_s, w_l)[-s:]
            )
            for w_s in range(2, 92)
            for w_l in range(2, 92)
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
#     print(p.calc_signal())


# asyncio.run(main())
