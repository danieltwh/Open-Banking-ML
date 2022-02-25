import os
import pandas as pd
import numpy as np
from technical_indicators import process


os.chdir("../")
SIGNAL_DIRECTORY = "data/processed/"
INPUT_FILENAME = "EUR_USD_Labelled_v1.csv"
data = pd.read_csv(SIGNAL_DIRECTORY + INPUT_FILENAME)
data = process(data)

# Save data
OUTPUT_DIRECTORY = "data/processed/signal"
data.to_csv(OUTPUT_DIRECTORY + INPUT_FILENAME[:-4] + "_processed.csv")
