#!/usr/bin/env python3
"""
visualize the pd.DataFrame

- The column Weighted_Price should be removed
- Rename the column Timestamp to Date
- Convert the timestamp values to date values
- Index the data frame on Date
- Missing values in High, Low, Open, and Close should
  be set to the previous row’s Close value
- Missing values in Volume_(BTC) and Volume_(Currency)
  should be set to 0
- Plot the data from 2017 and beyond at daily intervals
  and group the values of the same day such that:

High: max

Low: min

Open: mean

Close: mean

Volume_(BTC): sum

Volume_(Currency): sum 
"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
df = df.drop("Weighted_Price", axis=1)
df = df.rename(columns={"Timestamp": "Datetime"})
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
df.set_index("Datetime", inplace=True)
df["Close"] = df["Close"].fillna(method='backfill')
v = df["Close"]
df = df.fillna(value={"High": v, "Low" :v, "Open": v})
df = df.fillna(value={"Volume_(BTC)": 0, "Volume_(Currency)" :0})

df['2017-01-01 00:00:00'::].plot(kind='line', figsize=(8, 6))
plt.show()
