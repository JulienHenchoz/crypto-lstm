import sys
import datetime

import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

from utils.dataset import Dataset

file_name = sys.argv[1]

df = pd.read_csv(file_name, index_col=None)
df = df.sort_values('datetime')

frame_datetime = df.pop('datetime')
df = df.dropna()
min_max_scaler = MinMaxScaler()
y_min_max_scaler = MinMaxScaler()
df_array = np.array(df)
#df_array = df[0:20000]

scaled = min_max_scaler.fit_transform(df_array)
y_min_max_scaler.fit(df_array[:,0:1])

close_index = 0

do_train = True
# Number of features for prediction
features_count = df_array.shape[1]
num_units = 48
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(lr=learning_rate)
loss_function = 'mse'
batch_size = 256
num_epochs = 50

# Train on 80% of the dataset
train_split = int(df_array.shape[0] * 0.8)

dataset = Dataset()
x_train, y_train = dataset.prepare(scaled, 0, train_split, with_y=True)
x_test, y_test = dataset.prepare(scaled, train_split, None, with_y=True)

y_test_inverse = y_min_max_scaler.inverse_transform(y_test)

def train(x_train, y_train, x_test, y_test):
    # Initialize the RNN
    model = Sequential()
    model.add(LSTM(units=num_units, return_sequences=True, input_shape=(None, features_count)))
    model.add(Dropout(0.5))
    model.add(LSTM(units=num_units, return_sequences=True, input_shape=(None, features_count)))
    model.add(Dropout(0.5))
    model.add(LSTM(units=num_units))
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer=adam, loss=loss_function)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Using the training set to train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False,
        callbacks=[tensorboard_callback],
        validation_data=(x_test, y_test)
    )
    model.save('./models/latest')
    return model


model = train(x_train, y_train, x_test, y_test)

original = pd.DataFrame(y_test_inverse)
predictions = pd.DataFrame(y_min_max_scaler.inverse_transform(model.predict(x_test)))


print('Trend success : ' + str(dataset.get_trend_success_rate(predictions, original[0], df)))

figure(num=None, figsize=(32, 15), dpi=80, facecolor='w', edgecolor='k')
ax = sns.lineplot(x=original.index, y=original[0], label="Test Data", color='blue')
ax = sns.lineplot(x=predictions.index, y=predictions[0], label="Prediction latest", color='red')
ax.set_title('Bitcoin price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

plt.show()
