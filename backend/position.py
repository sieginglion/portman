import math
from typing import Literal, NamedTuple

import numba as nb
import numpy as np
from numpy import float64 as f8
from numpy.typing import NDArray as Array

from . import crypto, stock


def calc_k(w: int, e: float = 0.05):
    return math.ceil(math.log(e) / math.log(1 - 2 / (w + 1)))


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
    diff = np.diff(macd)
    macd = macd[1:]
    return ((macd < 0) & (diff > 0)).astype(f8) - ((macd > 0) & (diff < 0)).astype(f8)


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8], parts: int):
    cash, pos = 1000, 0
    left, side = parts, 1
    mid = len(prices) // 2 - 1
    for i in range(len(prices)):
        price, signal = prices[i], signals[i]
        if signal:
            if signal != side:
                left, side = parts, signal
            if left:
                size = (cash if side == 1 else pos * price) / left
                cash -= side * size
                pos += side * (size / price)
                left -= 1
        if i == mid:
            factor = 1000 / (cash + pos * price)
            cash *= factor
            pos *= factor
    return cash + pos * prices[-1]


class Best(NamedTuple):
    w_s: int
    w_l: int
    s: float


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
        m, n = scale // 91, scale * 2
        P = self.prices[-n:]
        best = Best(0, 0, 0)
        for w_s in range(m, scale + m, m):
            for w_l in range(m, scale + m, m):
                if w_s == 1 or w_s >= w_l:
                    continue
                S = gen_signals(self.prices, w_s, w_l)[-n:]
                for p in range(m, 8 * m, m):
                    if (s := simulate(P, S, p)) > best.s:
                        best = Best(w_s, w_l, s)
        return gen_signals(self.prices, best.w_s, best.w_l)[-scale:]


# import asyncio


# async def main():
#     p = await Position('c', 'ETH', 319)
#     print(p.calc_signals(91))


# if __name__ == '__main__':
#     asyncio.run(main())
