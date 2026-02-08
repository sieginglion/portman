import asyncio
from typing import Literal

import numpy as np
import pandas as pd
from httpx import AsyncClient

from . import shared, stock
from .shared import FMP_KEY, MARKET_TO_TIMEZONE


async def fetch_xps(symbol: str) -> pd.DataFrame:
    url = 'https://financialmodelingprep.com/stable/income-statement'
    params = {
        'apikey': FMP_KEY,
        'limit': 9,
        'period': 'quarter',
        'symbol': symbol,
    }
    async with AsyncClient() as client:
        data = (await client.get(url, params=params)).json()
    if len(data) != 9:
        raise ValueError
    df = pd.DataFrame(data).set_index('filingDate').sort_index()
    df['eps'] = df['epsDiluted'].rolling(4).sum()
    df['rps'] = (df['revenue'] / df['weightedAverageShsOutDil']).rolling(4).sum()
    return df[['eps', 'rps']].iloc[3:]


async def calc_px_score(market: Literal['t', 'u'], symbol: str) -> float:
    prices, xps = await asyncio.gather(
        stock.get_prices(market, symbol, 364, False),
        fetch_xps(symbol + shared.get_suffix(market, symbol)),
    )
    xps.index = pd.to_datetime(xps.index)
    xps = xps.reindex(
        pd.date_range(
            xps.index[0],
            pd.Timestamp.now(MARKET_TO_TIMEZONE[market]).normalize().tz_localize(None),
        ),
        method='ffill',
    ).tail(len(prices))
    if not (xps['eps'] > 0).all():
        raise ValueError

    def calc_score(s: pd.Series) -> float:
        l = np.log(s)
        u, d = l.mean(), l.std()
        return (l.iloc[-1] - (u - 2 * d)) / (4 * d)

    return (calc_score(prices / xps['eps']) + calc_score(prices / xps['rps'])) / 2


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
