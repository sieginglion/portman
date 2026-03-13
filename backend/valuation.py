import asyncio
from typing import Literal

import numpy as np
import pandas as pd
from httpx import AsyncClient

from . import shared
from .shared import FMP_KEY, add_suffix


async def fetch_xps(market: Literal['t', 'u'], symbol: str, q: int) -> pd.DataFrame:
    params = {
        'apikey': FMP_KEY,
        'limit': q + 5,
        'period': 'quarter',
    }
    if market == 't':
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ add_suffix(symbol) }'
        eps_col = 'epsdiluted'
    else:
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
        eps_col = 'epsDiluted'
    async with AsyncClient() as client:
        data = (await client.get(url, params=params)).json()
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    df['date'] = pd.to_datetime(df['date']) + pd.Timedelta(days=1)
    xps = pd.DataFrame(
        {
            'rps': (df['revenue'] / df['weightedAverageShsOutDil'])
            .rolling(4)
            .sum()
            .to_numpy(),
            'eps': df[eps_col].rolling(4).sum().to_numpy(),
        },
        df['date'],
    ).iloc[3:]
    return xps


async def calc_scores(
    market: Literal['t', 'u'], symbol: str, end_date: str, q: int
) -> tuple[float, float | None]:
    prices, xps = await asyncio.gather(
        shared.get_prices(market, symbol, 91 * (q + 2), False),
        fetch_xps(market, symbol, q),
    )
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=91 * q - 1)
    index = pd.date_range(end=end, periods=len(prices))
    df = pd.DataFrame({'price': prices}, index).join(xps).ffill().loc[start:end]
    if pd.isna(df['rps'].iloc[0]):
        raise ValueError

    def norm(m: pd.Series) -> float:
        l = np.log(m)
        lo, hi = l.quantile([0.011, 0.989])
        return (l.iloc[-1] - lo) / (hi - lo)

    return (
        norm(df['price'] / df['rps']),
        norm(df['price'] / df['eps']) if (df['eps'] > 0).all() else None,
    )


# def avg_by_period_ends(values, ends):
#     today = pd.Timestamp.now(tz='Asia/Taipei').normalize().tz_localize(None)
#     series = pd.Series(
#         values, index=pd.date_range(end=today, periods=len(values), freq='D')
#     )
#     ends = pd.to_datetime(ends)
#     starts = ends[:-1] + pd.Timedelta(days=1)
#     return pd.Series({e: series.loc[s:e].mean() for s, e in zip(starts, ends[1:])})


# async def calc_rps_from_finmind(symbol: str):
#     url = 'https://api.finmindtrade.com/api/v4/data'
#     params = {
#         'data_id': symbol,
#         'dataset': 'TaiwanStockFinancialStatements',
#         'start_date': arrow.now('Asia/Taipei')
#         .shift(days=-10 * 91)
#         .format('YYYY-MM-DD'),
#     }
#     headers = {'Authorization': f'Bearer {FINMIND_KEY}'}
#     async with AsyncClient() as client:
#         resp, fx = await asyncio.gather(
#             client.get(url, params=params, headers=headers), get_rates(client, 10 * 91)
#         )
#     df = (
#         pd.DataFrame(resp.json()['data'])
#         .pivot(index='date', columns='type', values='value')
#         .sort_index()
#         .iloc[-9:]
#     )
#     if len(df) != 9:
#         raise AssertionError
#     fx = avg_by_period_ends(fx, df.index)
#     df = df.iloc[1:]
#     df['Revenue'] /= fx.values
#     df['rps'] = (
#         (df['Revenue'] * df['EPS'] / df['EquityAttributableToOwnersOfParent'])
#         .rolling(4)
#         .sum()
#     )
#     return df['rps'].iloc[3:]
