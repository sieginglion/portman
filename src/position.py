import crypto
import stock
from shared import *


@nb.njit(parallel=True)
def calc_trends(prices: Array[f8], a: float, k: int) -> Array[f8]:
    ema = calc_ema(prices, a, k)
    return np.sign(ema[1:] - ema[:-1])


@nb.njit
def simulate(prices: Array[f8], trends: Array[f8]) -> float:
    last_trade, earned = 0, 0
    cash, position = 1000, 0
    last_price = 0
    for price, trend in zip(prices, trends):
        if price != last_price:
            if trend > 0 and cash > 0:
                last_trade = price
                cash, position = 0, cash / price
            elif trend < 0 and cash == 0:
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

    async def ainit(self):
        n = self.scale + 1
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

    def calc_metrics(self, theta: float = 0.0) -> Metrics:
        R = np.log(self.prices[1:] / self.prices[:-1]) - theta
        return {
            'ER': np.mean(R) + theta,
            'omega': np.sum(R[R > 0]) / -np.sum(R[R < 0]),
            'downside': (np.sum(R[R < 0] ** 2) / len(R)) ** 0.5,
        }

    def calc_signal(self) -> Literal[-1, 0, 1]:
        s, W = 182, [7, 14, 28]
        w_to_trends = {w: calc_trends(self.prices, 2 / (w + 1), calc_k(w)) for w in W}
        w_to_score = {w: simulate(self.prices[-s:], w_to_trends[w][-s:]) for w in W}
        w = max(w_to_score.items(), key=lambda x: x[1])[0]
        return w_to_trends[w][-1]


# async def main():
#     p = await Position('c', 'ETH', 364 * 2)
#     print(p.calc_metrics())
#     print(p.calc_signal())


# asyncio.run(main())
