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
    omega: f8
    downside: f8


class Position:
    def __init__(self, market: str, symbol: str, m_scale: int, max_w: int) -> None:
        self.market = market
        self.symbol = symbol
        self.m_scale = m_scale
        self.W = [7, 14, 28] + [
            2**i * 91 for i in range(int(math.log2(max_w / 91)) + 1)
        ]

    async def ainit(self):
        def calc_k(w: int) -> int:
            return round(math.log(0.1) / math.log(1 - 2 / (w + 1)))

        n = max(self.m_scale + 1, self.W[-1] * 2 + calc_k(self.W[-1]))
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)
        )
        if len(self.prices) < self.m_scale + 1:
            raise ValueError
        W = list(filter(lambda w: w * 2 + calc_k(w) <= len(self.prices), self.W))
        k = calc_k(W[-1])
        self.w_to_trends = {w: calc_trends(self.prices, 2 / (w + 1), k) for w in W}
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_metrics(self) -> Metrics:
        P = self.prices[-(self.m_scale + 1) :]
        R = np.log(P[1:] / P[:-1])
        D = R[R < 0]
        return {
            'omega': np.sum(R[R > 0]) / np.abs(np.sum(D)),
            'downside': (np.sum(D**2) / len(R)) ** 0.5,
        }

    def calc_signal(self, w_l: int) -> int:
        if w_l not in self.w_to_trends:
            return 0
        scale = w_l * 2
        w_s_to_score = {
            w_s: simulate_strategy(
                self.prices[-scale:],
                self.w_to_trends[w_l][-scale:],
                self.w_to_trends[w_s][-scale:],
            )
            for w_s in filter(lambda w: w < w_l, self.W)
        }
        w_s = max(w_s_to_score.items(), key=lambda x: x[1])[0]
        l_trend, s_trend = self.w_to_trends[w_l][-1], self.w_to_trends[w_s][-1]
        return int((s_trend - l_trend) / 2)


# async def main():
#     p = await Position('c', 'ETH', 364 * 4, 364 * 4)
#     print(p.calc_metrics())
#     print(p.calc_signal(364 * 2))


# asyncio.run(main())
