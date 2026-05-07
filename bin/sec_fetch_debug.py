#!/usr/bin/env python3
import argparse
import asyncio
import sys

import pandas as pd
from httpx import AsyncClient


URLS = {
    'eps': (
        'https://data.sec.gov/api/xbrl/companyconcept/'
        'CIK{}/us-gaap/EarningsPerShareDiluted.json',
        'USD/shares',
    ),
    'dsc': (
        'https://data.sec.gov/api/xbrl/companyconcept/'
        'CIK{}/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json',
        'shares',
    ),
}
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
    )
}


def dedupe_sec_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df['form'].isin({'20-F', '20-F/A'}).any():
        print(f'foreign filer: {sorted(set(df["form"].dropna()))}', file=sys.stderr)
        return pd.DataFrame()
    return df[['filed', 'val', 'start', 'end']].sort_values(
        'filed', kind='mergesort'
    ).drop_duplicates(['start', 'end'], keep='last')


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('fact', choices=['eps', 'dsc'])
    args = parser.parse_args()

    cik = sys.stdin.read().strip()
    if not cik:
        raise SystemExit('Expected CIK on stdin')
    url, unit = URLS[args.fact]
    async with AsyncClient(headers=HEADERS) as client:
        rows = (await client.get(url.format(cik.zfill(10)))).json()['units'][unit]
    dedupe_sec_rows(rows).to_csv(sys.stdout, index=False)


if __name__ == '__main__':
    asyncio.run(main())
