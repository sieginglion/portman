import position
from shared import *

app = FastAPI()


class Derived(TypedDict):
    score: f8
    signal: int


@app.get('/derived')
async def get_derived(market: str, symbol: str) -> Derived:
    p = await position.Position(market, symbol)
    return {'score': p.calc_score(), 'signal': p.calc_signal()}


@app.get('/margin')
async def get_margin(market: str, symbol: str) -> float:
    p = await position.Position(market, symbol, 91 * 16)
    signals = [p.calc_signal(s, True) for s in (p.W + [p.scale])[-4:]]
    return 0.75 + sum(signals) / len(signals) / 4


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
