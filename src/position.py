import crypto
import stock
from shared import *


def calc_signals(prices: Array[f8], w_s, w_l) -> Array[f8]:
    s_ema = calc_ema(prices, 2 / (w_s + 1), calc_k(w_s))
    l_ema = calc_ema(prices, 2 / (w_l + 1), calc_k(w_l))
    macd = s_ema[-len(l_ema) :] - l_ema
    slope = macd[1:] - macd[:-1]
    macd = macd[-len(slope) :]
    signals = np.zeros(len(slope))
    signals[(macd < 0) & (slope > 0)] = 1
    signals[(macd > 0) & (slope < 0)] = -1
    return signals


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8]) -> float:
    cash, position = 0, 0
    for i, price in enumerate(prices):
        if i and price != prices[i - 1]:
            value = position * price
            if value < 1000 and signals[i] > 0:
                cash -= 1000 - value
                position = 1000 / price
            elif value > 1000 and signals[i] < 0:
                cash += value - 1000
                position = 1000 / price
    return cash + value


class Metrics(TypedDict):
    ER: f8
    omega: f8
    downside: f8


class Signal(TypedDict):
    w_s: int
    w_l: int
    signal: f8


class Position:
    def __init__(
        self, market: Literal['c', 't', 'u'], symbol: str, m_scale: int, s_scale: int
    ) -> None:
        self.market = market
        self.symbol = symbol
        self.m_scale = m_scale
        self.s_scale = s_scale

    async def ainit(self):
        n = self.m_scale + 1  # TODO: take s_scale into consideration
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)
        )
        if len(self.prices) != n:
            raise ValueError
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_metrics(self, theta: float = 0) -> Metrics:
        R = np.log(self.prices[1:] / self.prices[:-1]) - theta
        return {
            'ER': np.mean(R) + theta,
            'omega': np.sum(R[R > 0]) / -np.sum(R[R < 0]),
            'downside': (np.sum(R[R < 0] ** 2) / len(R)) ** 0.5,
        }

    def calc_signal(self) -> Signal:
        s = self.s_scale + 1
        W_to_score = {
            (w_s, w_l): simulate(
                self.prices[-s:],
                calc_signals(self.prices, w_s, w_l)[-s:],
            )
            for w_s in range(7, 92, 7)
            for w_l in range(7, 92, 7)
            if w_s < w_l
        }
        (w_s, w_l), score = max(
            W_to_score.items(), key=lambda x: x[1] if x[1] else -INF
        )
        print((w_s, w_l), score)
        return {
            'w_s': w_s,
            'w_l': w_l,
            'signal': calc_signals(self.prices, w_s, w_l)[-1],
        }


# async def main():
#     p = await Position('u', 'MSFT', 364 * 2, 364)
#     print(p.calc_metrics())
#     print(p.calc_signal())


# asyncio.run(main())
