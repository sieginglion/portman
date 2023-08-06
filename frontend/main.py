from dataclasses import dataclass

import numpy as np
import requests as r
import streamlit as st
from numpy import float64 as f8
from numpy.typing import NDArray as Array
from plotly import graph_objects as go

BACKEND_HOST = st.text_input('BACKEND_HOST', 'localhost')


@dataclass
class Chart:
    P: list[float] | Array[f8]
    X_b: list[int]
    X_s: list[int]

    def __post_init__(self):
        self.P = np.array(self.P)


def get_charts(market: str, symbol: str):
    res = r.get(f'http://{BACKEND_HOST}:8080/charts?market={market}&symbol={symbol}')
    s, l = res.json()
    return Chart(*s), Chart(*l)


def plot_chart(c: Chart):
    fig = go.Figure(
        (
            go.Scatter(y=c.P),
            go.Scatter(x=c.X_b, y=c.P[c.X_b], mode='markers'),
            go.Scatter(x=c.X_s, y=c.P[c.X_s], mode='markers'),
        )
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


def visualize(market: str, symbol: str):
    charts = get_charts(market, symbol)
    plot_chart(charts[0])
    plot_chart(charts[1])


def main():
    market = st.selectbox('market', ('c', 't', 'u'))
    symbol = st.text_input('symbol', 'ETH')
    st.button('visualize', on_click=visualize, args=(market, symbol))


if __name__ == '__main__':
    main()
