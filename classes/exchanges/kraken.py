from datetime import datetime

import krakenex
import time
import ssl
import pandas as pd
import numpy as np

from classes.dataset import Dataset


class Kraken:
    def __init__(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    def fetch(self, timestamp, pair, hours_back, interval):
        data = np.array(self.get_raw_data(timestamp, pair, hours_back, interval))
        return self.to_dateframe(data, timestamp)

    """
    Get live data from the Kraken API for the given pair
    """
    @staticmethod
    def get_raw_data(timestamp, pair, hours_back, interval):
        seconds_back = int(hours_back) * 60 * 60
        from_timetamp = timestamp - seconds_back

        k = krakenex.API()
        return k.query_public('OHLC', {
            'pair': pair,
            'interval': interval,
            'since': from_timetamp
        })['result'][pair]

    """
    Convert raw data from Kraken to a machine learning edible dataframe
    """
    @staticmethod
    def to_dateframe(data, timestamp = None):
        columns = ['datetime', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        df = pd.DataFrame(data, columns=columns)
        df = df.astype(float)

        dataset = Dataset()
        df = dataset.add_indicators(df)

        df.dropna(inplace=True)
        df.pop('vwap')
        df.pop('count')
        df.pop('open')
        df.pop('high')
        df.pop('low')
        df.pop('volume')


        df['datetime'] = df.datetime.values.astype(np.int64) // 10 ** 9

        if timestamp:
            df = df[df['datetime'] < timestamp]

        return df

