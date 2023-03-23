import logging
from collections import namedtuple
from typing import Literal, TypedDict

import numba as nb
import numpy as np
from numpy import float64 as f8
from numpy.typing import NDArray as Array

import crypto
import stock
from shared import calc_ema, calc_k


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
    cash = 1000; position = 0  # fmt: skip
    last = prices[0]
    mid = len(prices) // 2
    for i in range(1, len(prices)):
        price, signal = prices[i], signals[i]
        if 0 < abs(price / last - 1) < 0.09 and signal:
            if signal == 1 and position == 0:
                position = cash / price
                cash = 0
            elif signal == -1 and position > 0:
                cash = position * price
                position = 0
        if i == mid:
            if cash:
                cash = 1000
            else:
                position = 1000 / price
        last = price
    return cash + position * prices[-1]


class Metrics(TypedDict):
    expected: f8
    omega: f8
    downside: f8


Signal = namedtuple('Signal', ['w_s', 'w_l', 'signal'])


class Position:
    def __init__(
        self, market: Literal['c', 't', 'u'], symbol: str, metric_scale: int
    ) -> None:
        self.market = market
        self.symbol = symbol
        self.metric_scale = metric_scale

    async def ainit(self):
        s = self.metric_scale + 1
        self.prices = await (
            crypto.get_prices(self.symbol, s)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, s)  # type: ignore
        )
        if len(self.prices) != s:
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

    def calc_signal(self, scale: int) -> Signal:
        s = scale * 2 + 1
        W_to_score = {
            (w_s, w_l): simulate(
                self.prices[-s:], calc_signals(self.prices, w_s, w_l)[-s:]
            )
            for w_s in range(2, scale + 1)
            for w_l in range(w_s + 1, scale + 1)
        }
        (w_s, w_l), score = max(W_to_score.items(), key=lambda x: x[1])
        logging.info((w_s, w_l, score))
        return Signal(w_s, w_l, calc_signals(self.prices, w_s, w_l)[-1])


# import asyncio


# async def main():
#     p = await Position('c', 'ETH', 364 * 2)
#     print(p.calc_metrics(0))
#     print(p.calc_signal(91))


# asyncio.run(main())
