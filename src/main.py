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
    signal: f8 | None


@app.get('/ranking')
def get_ranking(k: int) -> HTMLResponse:
    return HTMLResponse(ranking.get_ranking(k))


@app.get('/derived')
async def get_derived(
    market: Literal['c', 't', 'u'],
    symbol: str,
    m_scale: int,
    s_scale: int,
    theta: float,
) -> Derived:
    p = await position.Position(market, symbol, m_scale, s_scale)
    return {**p.calc_metrics(theta), 'signal': p.search_signal() if s_scale else None}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
