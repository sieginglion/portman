import crypto
import stock
from shared import *


@nb.njit(parallel=True)
def calc_trends(prices: A[f8], a: float, k: int) -> A[f8]:
    ema = calc_ema(prices, a, k)
    return np.sign(ema[1:] - ema[:-1])


@nb.njit
def simulate_position(prices: A[f8], s_trends: A[f8], l_trends: A[f8]) -> float:
    last_price = 0.0
    b_price, profit = 0.0, 0.0
    cash, units = 1000.0, 0.0
    for price, s_trend, l_trend in zip(prices, s_trends, l_trends):
        if price != last_price:
            if s_trend > 0 and l_trend < 0 and b_price == 0:
                b_price = price
                cash, units = 0, cash / price
            elif s_trend < 0 and l_trend > 0 and b_price != 0:
                profit, b_price = profit + units * (price - b_price), 0
                cash, units = units * price, 0
        last_price = price
    return profit


class Position:
    def __init__(self, market: str, symbol: str, n_iters: int) -> None:
        self.market = market
        self.symbol = symbol
        self.n_iters = n_iters
        self.k = round(math.log(0.1) / np.log(1 - 2 / (n_iters + 1)))

    async def ainit(self):
        n = self.n_iters + (self.k - 1) + 1
        self.prices = await (
            crypto.get_prices(self.symbol, n)
            if self.market == 'c'
            else stock.get_prices(self.market, self.symbol, n)
        )
        return self

    def __await__(self):
        return self.ainit().__await__()

    def calc_score(self) -> f8:
        prices = self.prices[-(self.n_iters + 1) :]
        returns = np.log(prices[1:] / prices[:-1])
        return np.sum(returns[returns > 0]) / np.sum(returns[returns < 0]) ** 2

    def calc_signal(self) -> f8:
        prices = self.prices[-self.n_iters :]
        window_to_trends = {
            window: calc_trends(self.prices, 2 / (window + 1), self.k)
            for window in range(7, self.n_iters + 1, 7)
        }
        windows_to_score = {}
        for s_window in range(7, int(self.n_iters / 2) + 1, 7):
            for l_window in range(s_window * 2, self.n_iters + 1, 7):
                windows_to_score[(s_window, l_window)] = simulate_position(
                    prices, window_to_trends[s_window], window_to_trends[l_window]
                )
        s_window, l_window = max(windows_to_score.items(), key=lambda x: x[1])[0]
        s_trend, l_trend = (
            window_to_trends[s_window][-1],
            window_to_trends[l_window][-1],
        )
        return s_trend if s_trend * l_trend < 0 else f8(0)


# async def main():
#     p = await Position('t', '2330', 365)
#     p.calc_score()
#     p.calc_signal()


# asyncio.run(main())
