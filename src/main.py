import position
import ranking
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


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
