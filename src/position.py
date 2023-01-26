import crypto
import stock
from shared import *


@nb.njit(parallel=True)
def calc_trends(prices: Array[f8], a: float, k: int) -> Array[f8]:
    ema = calc_ema(prices, a, k)
    return np.sign(ema[1:] - ema[:-1])


@nb.njit
def simulate_strategy(
    prices: Array[f8], l_trends: Array[f8], s_trends: Array[f8]
) -> float:
    last_trade, earned = 0, 0
    cash, position = 1000, 0
    last_price = 0
    for price, l_trend, s_trend in zip(prices, l_trends, s_trends):
        if price != last_price:
            if s_trend > 0 > l_trend and cash > 0:
                last_trade = price
                cash, position = 0, cash / price
            elif s_trend < 0 < l_trend and cash == 0:
                earned += position * (price - last_trade)
                cash, position = position * price, 0
        last_price = price
    return earned


class Metrics(TypedDict):
    ER: f8
    omega: f8
    downside: f8


class Position:
    def __init__(self, market: Literal['c', 't', 'u'], symbol: str, scale: int) -> None:
        self.market = market
        self.symbol = symbol
        self.scale = scale
        self.W = [7, 14, 28, 91, 182, 364]

    async def ainit(self):
        n = max(self.scale + 1, self.W[-1] * 2 + calc_k(self.W[-1]))
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)
        )
        if len(self.prices) != n:
            raise ValueError
        self.w_to_trends = {
            w: calc_trends(self.prices, 2 / (w + 1), calc_k(w)) for w in self.W
        }
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_metrics(self, theta: float = 0.0) -> Metrics:
        P = self.prices[-(self.scale + 1) :]
        R = np.log(P[1:] / P[:-1])
        u = np.mean(R)
        R -= theta
        return {
            'ER': u,
            'omega': np.sum(R[R > 0]) / -np.sum(R[R < 0]),
            'downside': (np.sum(R[R < 0] ** 2) / len(R)) ** 0.5,
        }

    def calc_signal(self, w_l: int) -> Literal[-1, 0, 1]:
        scale = w_l * 2
        w_s_to_score = {
            w_s: simulate_strategy(
                self.prices[-scale:],
                self.w_to_trends[w_l][-scale:],
                self.w_to_trends[w_s][-scale:],
            )
            for w_s in self.W
            if w_s < w_l
        }
        w_s = max(w_s_to_score.items(), key=lambda x: x[1])[0]
        l_trend, s_trend = self.w_to_trends[w_l][-1], self.w_to_trends[w_s][-1]
        return int((s_trend - l_trend) / 2)

    def calc_signals(self) -> list[Literal[-1, 0, 1]]:
        return [self.calc_signal(w_l) for w_l in self.W[3:]]


# async def main():
#     p = await Position('c', 'ETH', 364 * 4)
#     print(p.calc_metrics())
#     print(p.calc_signals())


# asyncio.run(main())
