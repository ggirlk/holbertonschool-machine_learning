#!/usr/bin/env python3
"""
script to fill in the missing data points in the pd.DataFrame:

- The column Weighted_Price should be removed
- missing values in Close should be set to the previous
  row’s Close
- missing values in High, Low, Open should be set to the
  previous row’s Close value
- missing values in Volume_(BTC) and Volume_(Currency)
  should be set to 0
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
df = df.drop(columns=["Weighted_Price"])
df["Close"] = df["Close"].fillna(method='backfill')
v = dfp["Close"]
df.fillna(value={"High": v, "Low" :v, "Open": v}, inplace=True)

print(df.head())
print(df.tail())
