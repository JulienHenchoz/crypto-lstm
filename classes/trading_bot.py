import datetime

from classes.dataset import Dataset


class TradingBot:
    def __init__(self, exchange, pair, interval, past_history, model_file):
        self.exchange = exchange
        self.pair = pair
        self.interval = interval
        self.past_history = past_history
        self.dataset = Dataset()
        self.model_file = model_file

    def predict_price(self, timestamp):
        # Get the timestamp of the last known point, 24h before the given one
        last_timestamp = int(timestamp - (60 * 60 * 24))

        df = self.exchange.fetch(
            timestamp=last_timestamp,
            pair=self.pair,
            hours_back=self.past_history * 2,
            interval=self.interval
        )
        # Only keep the last entries needed
        df = df.tail(self.past_history)

        predictions, y = self.dataset.predict(df, self.model_file, with_y=False)
        print(predictions)
        print(datetime.datetime.fromtimestamp(timestamp))
        exit()

    def tick(self):
        return
