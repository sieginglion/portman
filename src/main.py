import position
import ranking
from shared import *

app = FastAPI()


class Derived(TypedDict):
    omega: f8
    downside: f8
    signal: int


@app.get('/ranking')
def get_ranking(k: int) -> HTMLResponse:
    return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(market: str, symbol: str, metrics_scale: int) -> Derived:
    p = await position.Position(market, symbol, metrics_scale, 364)
    return {**p.calc_metrics(), 'signal': p.calc_signal(364, False)}


@app.get('/margin')
async def get_margin(market: str, symbol: str) -> float:
    p = await position.Position(market, symbol, 0, 364 * 8)
    signals = [p.calc_signal(s, True) for s in p.W[-3:] + [p.max_signal_scale]]
    return 0.75 + 0.25 * sum(signals) / len(signals)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
