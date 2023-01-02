from shared import *


def get_symbol_to_name_and_sector(k: int) -> dict[str, tuple[str, str]]:
    res = r.get(
        'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx',
        timeout=10,
    )
    return {
        r.Ticker.replace('.', '-'): (r.Name, r.Sector)
        for _, r in itertools.islice(
            pd.read_excel(io.BytesIO(res.content), header=4).iterrows(), k
        )
    }


def get_profits(symbol: str) -> Array[f8]:
    res = r.get(
        f'https://financialmodelingprep.com/api/v3/income-statement/{ symbol }',
        params={
            'apikey': FMP_KEY,
            'limit': 9,
            'period': 'quarter',
        },
        timeout=10,
    )
    time.sleep(0.2)
    res.raise_for_status()
    return np.array(
        [e['grossProfit'] for e in sorted(res.json(), key=lambda x: x['date'])[-8:]],
        f8,
    )


def get_symbol_to_profits(symbols: list[str]) -> dict[str, Array[f8]]:
    symbol_to_profits = {}
    for symbol in symbols:
        try:
            profits = get_profits(symbol)
        except r.HTTPError:
            logging.error(symbol)
        else:
            if len(profits) != 8:
                logging.warning(symbol)
            else:
                symbol_to_profits[symbol] = profits
    return symbol_to_profits


def calc_symbol_to_score(symbol_to_profits: dict[str, Array[f8]]) -> dict[str, float]:
    symbol_to_score = {}
    for symbol, profits in symbol_to_profits.items():
        ttm = np.convolve(profits, np.ones(4), 'valid')
        symbol_to_score[symbol] = (ttm[4] - ttm[0]).item()
    return symbol_to_score


def gen_ranking(
    symbol_to_score: dict[str, float],
    symbol_to_name_and_sector: dict[str, tuple[str, str]],
) -> str:
    return pd.DataFrame(
        [
            [e[0], *symbol_to_name_and_sector[e[0]]]
            for e in sorted(symbol_to_score.items(), key=lambda x: x[1], reverse=True)
        ]
    ).to_html(header=False, index=False)


def get_ranking(k: int) -> str:
    symbol_to_name_and_sector = get_symbol_to_name_and_sector(k)
    symbol_to_profits = get_symbol_to_profits(list(symbol_to_name_and_sector.keys()))
    symbol_to_score = calc_symbol_to_score(symbol_to_profits)
    return gen_ranking(symbol_to_score, symbol_to_name_and_sector)


# print(get_ranking(10))
