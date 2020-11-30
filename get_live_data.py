import ssl
import sys
import time
from datetime import datetime

from classes.exchanges.kraken import Kraken
from classes.trading_bot import TradingBot
from utils.constants import PAST_HISTORY, INTERVAL, PAIR, MODEL_FILE

exchange =  Kraken()
trading_bot = TradingBot(
    exchange=exchange,
    pair=PAIR,
    interval=INTERVAL,
    past_history=PAST_HISTORY,
    model_file=MODEL_FILE
)

trading_bot.predict_price(timestamp=datetime.fromisoformat('2020-11-27 23:00:01').timestamp())


#df = df.tail(PAST_HISTORY)
#print(df)
exit()

predictions, y = dataset.predict(df, './models/reference_model', with_y=False)

print(dataset.get_trend_success_rate(predictions, y, df))
print(df)
print(predictions)