import math
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
    cash, pos = 1000, 0
    mid = len(prices) // 2 - 1
    for i in range(len(prices)):
        price, signal = prices[i], signals[i]
        if signal == 1 and pos == 0:
            cash, pos = 0, cash / price
        elif signal == -1 and pos > 0:
            cash, pos = pos * price, 0
        if i == mid:
            cash, pos = 1000 if cash else 0, 1000 / price if pos else 0
    return cash + pos * prices[-1]


class Position:
    def __init__(self, market: Literal['c', 't', 'u'], symbol: str, n: int):
        self.market = market
        self.symbol = symbol
        self.n = n

    async def init(self):
        self.prices = await (
            crypto.get_prices(self.symbol, self.n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, self.n)  # type: ignore
        )
        return self

    def __await__(self):
        return self.init().__await__()

    def calc_signals(self, scale: int):
        n = scale * 2
        P = self.prices[-n:]
        W_to_s = {
            (w_s, w_l): simulate(P, gen_signals(self.prices, w_s, w_l)[-n:])
            for w_s in range(2, scale)
            for w_l in range(w_s + 1, scale + 1)
        }
        (w_s, w_l), _ = max(W_to_s.items(), key=lambda x: x[1])
        return gen_signals(self.prices, w_s, w_l)[-scale:]


# import asyncio


# async def main():
#     p = await Position('u', 'MSFT', 365)
#     print(p.calc_signals(91))


# if __name__ == '__main__':
#     asyncio.run(main())
