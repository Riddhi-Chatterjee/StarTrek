#NOTE: Currently this loader has some issues with the abbreviations used for the stock names

import pandas as pd
import numpy as np
import schedule
import time
from os.path import exists
import threading
import yfinance as yf
from datetime import datetime

def extractDate(dt):
    return str(dt)[:10]

def loadPrice(stock_name, isStartOfDay):
    if not exists("datasets/1day_"+stock_name+".csv"):
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        stock_df = yf.download(stock_name, start, end)
        stock_df.to_csv("datasets/1day_"+stock_name+".csv")
    else:
        end = datetime.now()
        start = datetime(end.year, end.month, end.day)
        stock_df = yf.download(stock_name, start, end)
        with open("datasets/1day_"+stock_name+".csv", "a") as ds:
            stock_df = stock_df.reset_index(level=['Date'])
            stock_df['Date'] = stock_df['Date'].apply(extractDate)

            string = ",".join(str(v) for v in stock_df.values.tolist()[0])
            ds.write(string + '\n')

loadPrice('AAPL', True)