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
    last_price = 0.0
    cash, position = 1000.0, 0.0
    for price, l_trend, s_trend in zip(prices, l_trends, s_trends):
        if price != last_price:
            if s_trend > 0 > l_trend and cash > 0:
                cash, position = 0, cash / price
            elif s_trend < 0 < l_trend and cash == 0:
                cash, position = position * price, 0
        last_price = price
    return cash + position * last_price


class Metrics(TypedDict):
    omega: f8
    downside: f8


class Position:
    def __init__(self, market: str, symbol: str, scale: int = 364) -> None:
        self.market = market
        self.symbol = symbol
        self.scale = scale
        self.W = [7, 14, 28] + [91 * 2**i for i in range(int(math.log2(scale / 91)))]

    async def ainit(self):
        k = round(math.log(0.1) / math.log(1 - 2 / (self.W[-1] + 1)))
        n = self.scale + (k - 1) + 1
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)
        )
        self.w_to_trends = {w: calc_trends(self.prices, 2 / (w + 1), k) for w in self.W}
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_metrics(self) -> Metrics:
        d = 7
        P = self.prices[-(self.scale + d) :]
        I = np.arange(0, len(P), d)
        P = np.flip(np.flip(P)[I])
        R = np.log(P[1:] / P[:-1])
        D = R[R < 0]
        return {
            'omega': np.sum(R[R > 0]) / np.abs(np.sum(D)),
            'downside': (np.sum(D**2) / len(R)) ** 0.5,
        }

    def calc_signal(self, scale: int = 364, fixed: bool = False) -> int:
        W_to_score = {}
        for w_l in self.W[-1 if fixed else 1 :]:
            for w_s in filter(lambda x: x < w_l, self.W):
                W_to_score[(w_l, w_s)] = simulate_strategy(
                    self.prices[-scale:],
                    self.w_to_trends[w_l][-scale:],
                    self.w_to_trends[w_s][-scale:],
                )
        w_l, w_s = max(W_to_score.items(), key=lambda x: x[1])[0]
        l_trend, s_trend = self.w_to_trends[w_l][-1], self.w_to_trends[w_s][-1]
        return int((s_trend - l_trend) / 2)


# async def main():
#     p = await Position('t', '2330')
#     print(p.calc_metrics())
#     print(p.calc_signal())


# asyncio.run(main())
