import os
import pandas as pd
import numpy as np

"""
This script processes the raw currency exchange rate csv file by labelling each row with either
    0 (Rise) 
    1 (Neutral)
    2 (Fall) 
"""


def label_data(df, col_name):
    df["prev_" + col_name] = df[col_name].shift()
    df["prev_" + col_name].loc[0] = df["prev_" + col_name].loc[1]  # Replace NaN with first value
    df["class_" + col_name] = np.where(df[col_name] < df["prev_" + col_name], "Fall",
                                       np.where(df[col_name] > df["prev_" + col_name], "Rise", "Neutral"))
    return df

# Change directories
os.chdir("../")
# ToDo: Maybe add input file name as argument
SIGNAL_DIRECTORY = "data/raw/signal/"
# INPUT_FILENAME = "currency_exchange_rates_02-01-1995_-_02-05-2018.csv"
INPUT_FILENAME = "eurusd_hour.csv"
fx_data = pd.read_csv(SIGNAL_DIRECTORY + INPUT_FILENAME)
# Obtain the tickers/countries
keys = fx_data.columns[2:]
for key in keys:
    fx_data = label_data(fx_data, key)
# Save data
OUTPUT_DIRECTORY = "data/processed/signal"
fx_data.to_csv(OUTPUT_DIRECTORY + INPUT_FILENAME[:-4] + "_processed.csv")
