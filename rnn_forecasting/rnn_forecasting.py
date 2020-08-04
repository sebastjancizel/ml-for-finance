import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pandas_datareader import data
import os
import plotly.express as px


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

start_date = "2014-01-01"
end_date = "2020-07-01"
SRC_DATA_FILENAME = "goog_data.pkl"

try:
    goog_data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data = data.DataReader("GOOG", "yahoo", start_date, end_date)
    goog_data.to_pickle(SRC_DATA_FILENAME)


df = goog_data
df["Date"] = df.index

close_data = df["Close"].values
close_data = close_data.reshape((-1, 1))

split_percent = 0.80
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df["Date"][:split]
date_test = df["Date"][split:]

print(len(close_train))
print(len(close_test))

look_back = 15

train_generator = TimeseriesGenerator(
    close_train, close_train, length=look_back, batch_size=20
)
test_generator = TimeseriesGenerator(
    close_test, close_test, length=look_back, batch_size=1
)

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, activation="relu", input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))


import plotly.graph_objects as go


trace1 = go.Scatter(x=date_train, y=close_train, mode="lines", name="Data")
trace2 = go.Scatter(x=date_test, y=prediction, mode="lines", name="Prediction")
trace3 = go.Scatter(x=date_test, y=close_test, mode="lines", name="Ground Truth")
layout = go.Layout(
    title="Google Stock", xaxis={"title": "Date"}, yaxis={"title": "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()
