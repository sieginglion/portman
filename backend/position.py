import logging
import math
from dataclasses import dataclass
from typing import Literal

import numba as nb
import numpy as np
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import crypto, stock


def calc_k(w: int):
    return math.ceil(math.log(0.05) / math.log(1 - 2 / (w + 1)))


@nb.njit
def calc_ema(arr: Array[f8], a: float, k: int):
    W = a * (1 - a) ** np.arange(k - 1, -1, -1)
    ema = np.empty(len(arr) - k + 1)
    for i in range(len(arr) - k + 1):
        ema[i] = np.sum(arr[i : i + k] * W)
    return ema


def gen_signals(prices: Array[f8], w_s: int, w_l: int):
    s_ema = calc_ema(prices, 2 / (w_s + 1), calc_k(w_s))
    l_ema = calc_ema(prices, 2 / (w_l + 1), calc_k(w_l))
    macd = s_ema[-len(l_ema) :] - l_ema
    diff = np.concatenate((np.zeros(1), np.diff(macd)))
    return ((macd < 0) & (diff > 0)).astype(f8) - ((macd > 0) & (diff < 0)).astype(f8)


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8]):
    cash, position = 1000, 0
    last, mid = prices[0], len(prices) // 2
    for i in range(1, len(prices)):
        price, signal = prices[i], signals[i]
        if 0 < abs(price / last - 1) < 0.09 and signal:
            diff = 1000 - position * price
            if np.sign(diff) == signal:
                cash -= diff
                position = 1000 / price
        if i == mid:
            factor = 1000 / (cash + position * price)
            cash *= factor
            position *= factor
        last = price
    return cash + position * prices[-1]


@dataclass
class Signals:
    w_s: int
    w_l: int
    values: Array[f8]


class Position:
    def __init__(self, market: Literal['c', 't', 'u'], symbol: str, n: int):
        self.market = market
        self.symbol = symbol
        self.n = n

    async def ainit(self):
        self.prices = await (
            crypto.get_prices(self.symbol, self.n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, self.n)  # type: ignore
        )
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_score(self, scale: int):
        P = self.prices[-(scale + 1) :]
        R = np.log(P[1:] / P[:-1])
        return np.sum(R[R > 0]) / -np.sum(R[R < 0]) / np.std(R)

    def calc_signals(self, scale: int):
        n = scale * 2 + 1
        P = self.prices[-n:]
        W_to_s = {
            (w_s, w_l): simulate(P, gen_signals(self.prices, w_s, w_l)[-n:])
            for w_s in range(2, scale + 1)
            for w_l in range(w_s + 1, scale + 1)
        }
        (w_s, w_l), s = max(W_to_s.items(), key=lambda x: x[1])
        logging.info(s)
        return Signals(w_s, w_l, gen_signals(self.prices, w_s, w_l)[-scale:])


# import asyncio

# logging.basicConfig(level=logging.INFO)


# async def main():
#     p = await Position('u', 'MSFT', 319)
#     print(p.calc_score(364))
#     print(p.calc_signals(91))


# if __name__ == '__main__':
#     asyncio.run(main())
