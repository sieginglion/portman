import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import requests as r
import streamlit as st
from numpy import float64 as f8
from numpy.typing import NDArray as Array
from plotly import graph_objects as go

BACKEND_HOST = os.getenv('BACKEND_HOST', 'localhost')


@dataclass
class Chart:
    P: Array[f8]
    X_b: list[int]
    X_s: list[int]

    def __post_init__(self):
        self.P = np.array(self.P)


def get_charts(market: Literal['c', 't', 'u'], symbol: str):
    res = r.get(f'http://{BACKEND_HOST}:8080/charts?market={market}&symbol={symbol}')
    short, long = res.json()
    return Chart(*short), Chart(*long)


def plot_chart(chart: Chart):
    st.plotly_chart(
        go.Figure(
            (
                go.Scatter(y=chart.P),
                go.Scatter(x=chart.X_b, y=chart.P[chart.X_b], mode='markers'),
                go.Scatter(x=chart.X_s, y=chart.P[chart.X_s], mode='markers'),
            ),
            {'showlegend': False},
        )
    )


def main():
    market = st.selectbox('Market', ('c', 't', 'u'))
    symbol = st.text_input('Symbol', 'ETH')
    if st.button('Run'):
        short, long = get_charts(market, symbol)
        plot_chart(short)
        plot_chart(long)


if __name__ == '__main__':
    main()
