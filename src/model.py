import torch.nn as nn


"""
Macroeconomic indicator LSTM
Input factors: 
Interest Rate
Federal Reserves fund rates for the US
Inflation rate EU
Inflation rate US
S&P 500 Close value
DAX Close value 
"""

class MacroLSTM(nn.Module):
    def __init__(self):
        pass