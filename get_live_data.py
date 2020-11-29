import ssl
import sys
import time
import urllib.request, json
import pandas as pd
import krakenex

from utils.dataset import add_indicators, predict

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get_raw_data(pair, hours_back, interval = 60):

    seconds_back = int(hours_back) * 60 * 60
    from_timetamp = int(time.time()) - seconds_back
    k = krakenex.API()
    return k.query_public('OHLC', {
        'pair': pair,
        'interval': 60,
        'since': from_timetamp
    })['result'][pair]

pair = sys.argv[1]
hours_back = sys.argv[2]
ohlc = get_raw_data(pair, hours_back)

columns = ['datetime', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
df = pd.DataFrame(ohlc, columns = columns)
df = df.astype(float)

df = add_indicators(df)
df.dropna(inplace=True)
df.pop('vwap')
df.pop('count')
df.pop('open')
df.pop('high')
df.pop('low')
df.pop('volume')

predictions, y = predict(df, './reference_model', with_y=False)

print(df)
print(predictions)