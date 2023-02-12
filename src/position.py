import crypto
import stock
from shared import *


def calc_signals(prices: Array[f8], w_s, w_l) -> Array[f8]:
    s_ema = calc_ema(prices, 2 / (w_s + 1), calc_k(w_s))
    l_ema = calc_ema(prices, 2 / (w_l + 1), calc_k(w_l))
    macd = s_ema[-len(l_ema) :] - l_ema
    slow = calc_ema(macd, 2 / (7 + 1), calc_k(7))
    macd = macd[-len(slow) :]
    signals = np.zeros(len(macd))
    signals[(macd < 0) & (macd > slow)] = 1
    signals[(macd > 0) & (macd < slow)] = -1
    return signals


@nb.njit
def simulate(prices: Array[f8], signals: Array[f8]) -> float:
    cost, profit = 0, 0
    units = 0
    for i, price in enumerate(prices):
        if i and price != prices[i - 1]:
            value = units * price
            if value < 1000 and signals[i] > 0:
                cost += 1000 - value
                units = 1000 / price
            elif value > 1000 and signals[i] < 0:
                profit += (value - 1000) - cost
                cost = 0
                units = 1000 / price
    return profit


class Metrics(TypedDict):
    ER: f8
    omega: f8
    downside: f8


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

    def search_signal(self) -> f8:
        s, W = self.s_scale + 1, [7, 14, 28]
        W_to_score = {
            (w_s, w_l): simulate(
                self.prices[-s:],
                calc_signals(self.prices, w_s, w_l)[-s:],
            )
            for w_s in W
            for w_l in W
            if w_s < w_l
        }
        (w_s, w_l), score = max(W_to_score.items(), key=lambda x: x[1])
        if not score:
            raise ValueError
        return calc_signals(self.prices, w_s, w_l)[-1]


# async def main():
#     p = await Position('u', 'MSFT', 364 * 2, 182)
#     print(p.calc_metrics())
#     print(p.search_signal())


# asyncio.run(main())
