import datetime

import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
import pandas as pd

from utils.constants import PAST_HISTORY, PREDICT_FORWARD, CLOSE_INDEX

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class Dataset:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        self.y_min_max_scaler = MinMaxScaler()

    def predict(self, df, model_file, with_y=True):
        df = df.dropna()
        df_x = df.copy()
        df_x.pop('datetime')
        df_x = np.array(df_x)
        df_y = df['close']

        df_y = np.array(df_y)

        scaled = self.min_max_scaler.fit_transform(df_x)
        self.y_min_max_scaler.fit(df_y.reshape(-1, 1))

        X, y = self.prepare(scaled, 0, df_x.shape[0], with_y)

        model_reference = load_model(model_file)
        predictions = pd.DataFrame(self.y_min_max_scaler.inverse_transform(model_reference.predict(X)))

        y = self.y_min_max_scaler.inverse_transform(y).reshape(1, -1)[0]

        return predictions, y

    @staticmethod
    def add_indicators(df):
        df['rsi'] = talib.RSI(np.array(df['close']))
        df['adx'] = talib.ADX(np.array(df['high']), np.array(df['low']), np.array(df['close']))
        df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(np.array(df['close']))
        df['macd'], df['macd_sig'], _ = talib.MACD(np.array(df['close']))
        df['boll_up'], df['boll_mid'], df['boll_low'] = talib.BBANDS(df['close'])
        df['datetime'] = df['datetime'].apply(datetime.datetime.fromtimestamp)
        return df

    @staticmethod
    def prepare(dataset, start_index, end_index, with_y=True):
        data = []
        labels = []
        start_index = start_index + PAST_HISTORY
        features_count = dataset.shape[1]

        if end_index is None:
            end_index = len(dataset)

        range_max = end_index - PREDICT_FORWARD if with_y else end_index + 1

        for i in range(start_index, range_max):
            # Label is the price at t+X
            if with_y:
                y = dataset[i + PREDICT_FORWARD][CLOSE_INDEX]
            else:
                y = None
            X = dataset[i - PAST_HISTORY:i]

            data.append(np.reshape(X, (PAST_HISTORY, features_count)))
            labels.append([y])

        return np.array(data), np.array(labels)

    @staticmethod
    def mean_error(predictions, reality):
        c = []
        for i in range(len(reality)):
            c.append(float(predictions[i] - reality[i]) * 100 / reality[i])
        return np.mean(c)

    @staticmethod
    def get_trend_success_rate(predictions, reality, df):
        successes = 0

        for i in range(len(reality)):
            predicted_value = predictions.values[i][0]
            future_value = reality[i]
            initial_price = df['close'].values[i]
            if (predicted_value > initial_price and future_value > initial_price) or (predicted_value < initial_price and future_value < initial_price):
                successes += 1


        return float(successes * 100 / len(reality))