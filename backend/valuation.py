import asyncio
from typing import Literal

import numpy as np
import pandas as pd
from httpx import AsyncClient

from . import shared, stock
from .shared import FMP_KEY, MARKET_TO_TIMEZONE


async def fetch_xps(market: Literal['t', 'u'], symbol: str, q: int) -> pd.DataFrame:
    params = {
        'apikey': FMP_KEY,
        'limit': q + 5,
        'period': 'quarter',
    }
    if market == 't':
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ shared.add_suffix(symbol) }'
        date_col = 'date'
        date_offset = 1
        eps_col = 'epsdiluted'
    else:
        url = 'https://financialmodelingprep.com/stable/income-statement'
        params['symbol'] = symbol
        date_col = 'filingDate'
        date_offset = 0
        eps_col = 'epsDiluted'
    async with AsyncClient() as client:
        data = (await client.get(url, params=params)).json()
    df = pd.DataFrame(data).sort_values(date_col)
    return pd.DataFrame(
        {
            'eps': df[eps_col].rolling(4).sum().to_numpy(),
            'rps': (df['revenue'] / df['weightedAverageShsOutDil'])
            .rolling(4)
            .sum()
            .to_numpy(),
        },
        pd.to_datetime(df[date_col]) + pd.Timedelta(days=date_offset),
    ).iloc[3:]


async def calc_px_score(market: Literal['t', 'u'], symbol: str, q: int) -> float:
    prices, xps = await asyncio.gather(
        stock.get_prices(market, symbol, 91 * q, False), fetch_xps(market, symbol, q)
    )
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
