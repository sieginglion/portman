import position
import ranking
from shared import *

app = FastAPI()


class Derived(TypedDict):
    omega: f8
    downside: f8
    signals: list[int]


@app.get('/ranking')
def get_ranking(k: int) -> HTMLResponse:
    return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(market: str, symbol: str, m_scale: int, max_w: int) -> Derived:
    p = await position.Position(market, symbol, m_scale, max_w)
    return {**p.calc_metrics(), 'signals': [p.calc_signal(w_l) for w_l in p.W[3:]]}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
