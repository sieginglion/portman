import crypto
import position
import ranking
import stock
from shared import *

app = FastAPI()


class Derived(TypedDict):
    ER: f8
    omega: f8
    downside: f8
    signals: list[Literal[-1, 0, 1]]


@app.get('/ranking')
def get_ranking(k: int) -> HTMLResponse:
    return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(
    market: Literal['c', 't', 'u'], symbol: str, scale: int, theta: float = 0.0
) -> Derived:
    p = await position.Position(market, symbol, scale)
    return {**p.calc_metrics(theta), 'signals': p.calc_signals()}


@app.get('/margin')
async def get_margin(market: Literal['c', 't', 'u'], symbol: str) -> f8:
    w = 364 * 8
    k = calc_k(w)
    n = 364 * 4 + k - 1
    P = await (
        crypto.get_prices(symbol, n)
        if market == 'c'
        else stock.get_prices(market, symbol, n)
    )
    ema = calc_ema(P, 2 / (w + 1), k)
    M = 1 - P[-len(ema) :] / ema
    p50, p99 = np.percentile(M, 50), np.percentile(M, 99)
    M = 10 / 12 + 2 / 12 * (M - p50) / (p99 - p50)
    return M[-1]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
