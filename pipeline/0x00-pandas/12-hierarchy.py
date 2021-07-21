#!/usr/bin/env python3
"""
rearrange the MultiIndex levels such
that timestamp is the first level:

- Concatenate th bitstamp and coinbase tables from
  timestamps 1417411980 to 1417417980, inclusive
- Add keys to the data labeled bitstamp and coinbase
  respectively
- Display the rows in chronological order
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')


df = pd.concat([df2.iloc[1417411980:1417417980].set_index(["Timestamp"]),
                df1.iloc[1417411980:1417417980].set_index(["Timestamp"])],
               keys=['bitstamp', 'coinbase'])
df = df.swaplevel().sort_index()

print(df)
