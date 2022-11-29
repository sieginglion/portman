import crypto
import position
import stock
from shared import *

app = FastAPI()


@app.get('/derived')
async def get_derived(market: str, symbol: str) -> Dict[str, f8]:
    n_iters = 365
    p = await position.Position(market, symbol, n_iters)
    return {'score': p.calc_score(), 'signal': p.calc_signal()}


@app.get('/margin')
async def get_margin(market: str) -> f8:
    symbol = {'t': 'TW50', 'u': 'SPY', 'c': 'BTC'}[market]
    n_iters = 365 * 4
    a = 2 / (365 * 8 + 1)
    k = round(math.log(0.1) / math.log(1 - a))
    n = n_iters + (k - 1)
    prices = await (
        crypto.get_prices(symbol, n)
        if market == 'c'
        else stock.get_prices(market, symbol, n)
    )
    ema = calc_ema(prices, a, k)
    return ema[-1] / prices[-1]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
