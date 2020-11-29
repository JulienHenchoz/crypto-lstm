from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import numpy as np
import pandas as pd
import matplotlib
import talib

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    file_name = sys.argv[1]
    df = pd.read_csv(
        file_name,
        sep=',',
        usecols=['Unix Timestamp','Close', 'High', 'Low', 'Volume XRP']
    )
    df.rename(
        columns={
            "Unix Timestamp": "datetime",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume XRP": "volume",
        },
        inplace=True
    )
    df['datetime'] = df['datetime'].apply(datetime.datetime.fromtimestamp)
    df = df.set_index('datetime')
    df = df.sort_values(by='datetime', ascending=True)

    df['rsi'] = talib.RSI(np.array(df['close']))
    df['adx'] = talib.ADX(np.array(df['high']), np.array(df['low']), np.array(df['close']))
    df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(np.array(df['close']))
    df['macd'], df['macd_sig'], _ = talib.MACD(np.array(df['close']))
    df['boll_up'], df['boll_mid'], df['boll_low'] = talib.BBANDS(df['close'])
    #exit()

    # Add a strategy
    output_file = os.path.splitext(os.path.basename(file_name))[0] + '_indicators.csv'
    output_path = os.path.dirname(file_name)
    full_path = output_path + '/' + output_file


    df.to_csv(full_path, columns=['close', 'rsi', 'stochrsi_k', 'stochrsi_d', 'adx', 'boll_up', 'boll_mid', 'boll_low' ,'macd', 'macd_sig'])