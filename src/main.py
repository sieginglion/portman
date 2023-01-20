import crypto
import position
import ranking
import stock
from shared import *

app = FastAPI()


class Derived(TypedDict):
    omega: f8
    downside: f8
    signals: list[Literal[-1, 0, 1]]


@app.get('/ranking')
def get_ranking(k: int) -> HTMLResponse:
    return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(market: str, symbol: str, m_scale: int, max_w: int) -> Derived:
    p = await position.Position(market, symbol, m_scale, max_w)
    return {**p.calc_metrics(), 'signals': p.calc_signals()}


@app.get('/margin')
async def get_margin(market: str, symbol: str) -> float:
    w = 364 * 4
    k = calc_k(w)
    n = w * 2 + (k - 1)
    prices = await (
        crypto.get_prices(symbol, n)
        if market == 'c'
        else stock.get_prices(market, symbol, n)
    )
    ema = calc_ema(prices, 2 / (w + 1), k)
    ratios = ema / prices[-len(ema) :]
    p50, p95 = np.percentile(ratios, 50), np.percentile(ratios, 95)
    margins = (ratios - p50) / (p95 - p50) * 0.1 + 0.9
    logging.info(min(margins), max(margins))
    return margins[-1].item()


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
