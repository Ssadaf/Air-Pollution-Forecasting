#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import os
import cv2 
from PIL import Image 
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Input
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from time import time
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import load
from sklearn.preprocessing import MinMaxScaler
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from random import randint
from keras.initializers import RandomNormal
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


data = load('gdrive/My Drive/MiniProj2/polution_dataSet.npy')


# In[ ]:


# data_scale = MinMaxScaler()
# data_s = data_scale.fit_transform(data)

train_data, test_data = data[:12000], data[12000:15000]


# In[ ]:


def create_dataset(dataset, window_size=1, stride=1):
	dataX, dataY = [], []
	for i in range(0, len(dataset)-window_size-1, stride):
		dataX.append(dataset[i:(i + window_size)])
		dataY.append(dataset[i + window_size][0])
	return np.array(dataX), np.array(dataY)


# In[7]:


x_train, y_train = create_dataset(train_data, 11)
x_test, y_test = create_dataset(test_data, 11)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[ ]:


def plot_prediction(predicted, actual):
  plt.figure(figsize=(15,8))
  plt.plot(predicted)
  plt.plot(actual, linestyle='-')
  plt.legend(['predicted', 'actual'])
  plt.xlabel("Time")
  plt.ylabel("Pollution")
  plt.show()


# In[ ]:


def show_train_loss_history(trained, title=None, show_validation=True):
    plt.plot(trained.history['loss'])
    if(show_validation):
      plt.plot(trained.history['val_loss'])      
      plt.legend(['train_loss', 'validation_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if(title):
      plt.title(title)
    plt.show()


# # Simple RNN

# In[ ]:


model = Sequential([
    SimpleRNN(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])


# In[ ]:


start=time()
trained_rnn = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)


# In[ ]:


predicted = trained_rnn.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]


# In[ ]:


actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse) 
plot_prediction(wanted, actual)
show_train_loss_history(trained_rnn, "Training Loss Plot")


# # LSTM

# In[ ]:


model = Sequential([
    LSTM(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])


# In[ ]:


start=time()
trained_lstm = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)


# In[ ]:


predicted = trained_lstm.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]


# In[ ]:


actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse) 
plot_prediction(wanted, actual)
show_train_loss_history(trained_lstm, "Training Loss Plot")


# # GRU

# In[ ]:


model = Sequential([
    GRU(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])


# In[ ]:


start=time()
trained_gru = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)


# In[ ]:


predicted = trained_gru.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]


# In[ ]:


actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse) 
plot_prediction(wanted, actual)
show_train_loss_history(trained_gru, "Training Loss Plot")


# # Optimizers and Loss Functions

# ## MSE

# ### Optimizers for Simple RNN with MSE Loss function

# In[ ]:


trained_rnn_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:

  model = Sequential([
    SimpleRNN(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])
  
  print("###", opt, "###")
  model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_rnn_opts.append(trained_opt)


# ### Optimizers for LSTM model with MSE Loss function

# In[ ]:


trained_lstm_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:
  model = Sequential([
    LSTM(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])

  print("###", opt, "###")
  model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_lstm_opts.append(trained_opt)


# ### Optimizers for GRU model with MSE Loss function

# In[ ]:


trained_gru_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:
  model = Sequential([
    GRU(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])

  print("###", opt, "###")
  model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_gru_opts.append(trained_opt)


# ## MAE

# ### Optimizers for Simple RNN with MAE Loss function

# In[ ]:


trained_rnn_mae_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:

  model = Sequential([
    SimpleRNN(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])
  
  print("###", opt, "###")
  model.compile(loss='mae', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_rnn_mae_opts.append(trained_opt)


# ### Optimizers for LSTM model with MAE Loss function

# In[ ]:


trained_lstm_mae_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:
  model = Sequential([
    LSTM(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])

  print("###", opt, "###")
  model.compile(loss='mae', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_lstm_mae_opts.append(trained_opt)


# ### Optimizers for GRU model with MAE Loss function

# In[ ]:


trained_gru_mae_opts = []

for opt in ['adam', 'RMSprop', 'Adagrad']:
  model = Sequential([
    GRU(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
  ])

  print("###", opt, "###")
  model.compile(loss='mae', optimizer=opt, metrics=['mse', 'mae'])

  start=time()
  trained_opt = model.fit(x_train, y_train, epochs=20, shuffle=True, validation_split=0.2, batch_size=10)
  train_time = time()-start
  print("Training time:", train_time)

  predicted = trained_opt.model.predict(x_test)
  wanted = [predicted[i] for i in range(0, len(predicted), 12)]
  actual = [y_test[i] for i in range(0, len(predicted), 12)]
  mse = mean_squared_error(actual, wanted) 
  print("MSE for", opt, "= ", mse)
  plot_prediction(wanted, actual)
  show_train_loss_history(trained_opt, "Training Loss Plot for" + opt)
  trained_gru_mae_opts.append(trained_opt)


# # Different Time Series

# In[ ]:


larger_train_data, larger_test_data = data[:18000], data[18000:22800]


# ## Same Hour Every Day

# In[ ]:


random_hour = randint(0,23)
train_data_per_day = [larger_train_data[i] for i in range(random_hour, len(larger_train_data), 24)]
test_data_per_day = [larger_test_data[i] for i in range(random_hour, len(larger_test_data), 24)]


# In[ ]:


x_train_per_day, y_train_per_day = create_dataset(train_data_per_day, 6)
x_test_per_day, y_test_per_day = create_dataset(test_data_per_day, 6)
print(x_train_per_day.shape, y_train_per_day.shape)
print(x_test_per_day.shape, y_test_per_day.shape)


# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(6, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_per_day = model.fit(x_train_per_day, y_train_per_day, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_per_day.model.predict(x_test_per_day)

mse = mean_squared_error(y_test_per_day, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_per_day)
show_train_loss_history(trained_per_day, "Training Loss", False)


# ## Same Hour and Day Every Week

# In[ ]:


random_day = randint(0,7)
random_hour_in_day = randint(0,23)
train_data_per_week = [larger_train_data[i] for i in range(random_day*24 + random_hour_in_day, len(larger_train_data), 7*24)]
test_data_per_week = [larger_test_data[i] for i in range(random_day*24 + random_hour_in_day, len(larger_test_data), 7*24)]


# In[ ]:


x_train_per_week, y_train_per_week = create_dataset(train_data_per_week, 3)
x_test_per_week, y_test_per_week = create_dataset(test_data_per_week, 3)
print(x_train_per_week.shape, y_train_per_week.shape)
print(x_test_per_week.shape, y_test_per_week.shape)


# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(3, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_per_week = model.fit(x_train_per_week, y_train_per_week, epochs=200, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_per_week.model.predict(x_test_per_week)

mse = mean_squared_error(y_test_per_week, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_per_week)
show_train_loss_history(trained_per_week, "Training Loss", False)


# # Dropout

# ## Without Dropout

# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_no_dropout = model.fit(x_train, y_train, epochs=100, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_no_dropout.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]
actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse)
plot_prediction(wanted, actual)
show_train_loss_history(trained_no_dropout, "Training Loss Plot")


# ## With Dropout

# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(11, 8)),
  Dropout(0.1, seed=5),
  Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_dropout = model.fit(x_train, y_train, epochs=100, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_dropout.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]
actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse)
plot_prediction(wanted, actual)
show_train_loss_history(trained_dropout, "Training Loss Plot")


# # Fusion

# ## Preparing Data

# In[ ]:


larger_train_data, larger_test_data = data[:18000], data[18000:22800]


# In[ ]:


day = 1
hour_in_day = 16

train_data_daily = [larger_train_data[i] for i in range((day+1)*24 + hour_in_day, len(larger_train_data), 24)]
test_data_daily = [larger_test_data[i] for i in range((day+1)*24 + hour_in_day, len(larger_test_data), 24)]

train_data_weekly = [larger_train_data[i] for i in range(day*24 + hour_in_day, len(larger_train_data), 7*24)]
test_data_weekly = [larger_test_data[i] for i in range(day*24 + hour_in_day, len(larger_test_data), 7*24)]


# In[ ]:


x_train_hourly, y_train_hourly = create_dataset(larger_train_data[(day)*24 + (hour_in_day-11):], 11, 24*7)
x_test_hourly , y_test_hourly = create_dataset(larger_test_data[(day)*24 + (hour_in_day-11):], 11, 24*7)
print(x_train_hourly.shape, y_train_hourly.shape)
print(x_test_hourly.shape, y_test_hourly.shape)

x_train_daily, y_train_daily = create_dataset(train_data_daily, 6, 7)
x_test_daily, y_test_daily = create_dataset(test_data_daily, 6, 7)
print(x_train_daily.shape, y_train_daily.shape)
print(x_test_daily.shape, y_test_daily.shape)

x_train_weekly, y_train_weekly = create_dataset(train_data_weekly, 3)
x_test_weekly, y_test_weekly = create_dataset(test_data_weekly, 3)
print(x_train_weekly.shape, y_train_weekly.shape)
print(x_test_weekly.shape, y_test_weekly.shape)


# In[ ]:


num_train_data = len(y_train_weekly)
x_train_hourly, y_train_hourly = x_train_hourly[3:3+num_train_data], y_train_hourly[3:3+num_train_data]
print(x_train_hourly.shape, y_train_hourly.shape)
x_train_daily, y_train_daily = x_train_daily[2:2+num_train_data], y_train_daily[2:2+num_train_data]
print(x_train_daily.shape, y_train_daily.shape)
print(x_train_weekly.shape, y_train_weekly.shape)

num_test_data = len(y_test_weekly)
x_test_hourly, y_test_hourly = x_test_hourly[3:3+num_test_data], y_test_hourly[3:3+num_test_data]
print(x_test_hourly.shape, y_test_hourly.shape)
x_test_daily, y_test_daily = x_test_daily[2:2+num_test_data], y_test_daily[2:2+num_test_data]
print(x_test_daily.shape, y_test_daily.shape)
print(x_test_weekly.shape, y_test_weekly.shape)


# ## Training Model

# In[ ]:


model_hourly = Sequential([
  LSTM(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model_hourly.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_hourly = model_hourly.fit(x_train_hourly, y_train_hourly, epochs=10, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted_hourly = model_hourly.predict(x_test_hourly)
mse = mean_squared_error(y_test_hourly, predicted_hourly) 
print("MSE = ", mse)


# In[ ]:


model_daily = Sequential([
  LSTM(100, input_shape=(6, 8)),
  Dense(1, activation='tanh')
])

model_daily.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_daily = model_daily.fit(x_train_daily, y_train_daily, epochs=10, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

from sklearn.metrics import mean_squared_error

predicted_daily = model_daily.predict(x_test_daily)
mse = mean_squared_error(y_test_daily, predicted_daily) 
print("MSE = ", mse)


# In[ ]:


model_weekly = Sequential([
  LSTM(50, input_shape=(3, 8)),
  Dense(1, activation='tanh')
])

model_weekly.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_weekly = model_weekly.fit(x_train_weekly, y_train_weekly, epochs=10, shuffle=True, validation_split=0.2, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

from sklearn.metrics import mean_squared_error

predicted_weekly = model_weekly.predict(x_test_weekly)
mse = mean_squared_error(y_test_weekly, predicted_weekly) 
print("MSE = ", mse)


# In[ ]:


pred1 = model_hourly.predict(x_train_hourly)
pred2 = model_daily.predict(x_train_daily)
pred3 = model_weekly.predict(x_train_weekly)

new_data_train = [[*pred1[i], *pred2[i], *pred3[i]] for i in range(len(pred1))]
new_data_train = np.array(new_data_train)


# In[ ]:


pred1 = model_hourly.predict(x_test_hourly)
pred2 = model_daily.predict(x_test_daily)
pred3 = model_weekly.predict(x_test_weekly)

new_data_test = [[*pred1[i], *pred2[i], *pred3[i]] for i in range(len(pred1))]
new_data_test = np.array(new_data_test)


# In[ ]:


model = Sequential([
  Dense(1, activation='relu',  input_shape=(3,), use_bias=False)
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained = model.fit(new_data_train, y_train_weekly, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = model.predict(new_data_test)
mse = mean_squared_error(y_test_weekly, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_weekly)
show_train_loss_history(trained, "Training Loss Plot", False)


# In[ ]:


print(model.weights)


# # Selecting Two Features

# ## Preparing Data

# In[ ]:


def corr(f, g):
    num, den1 ,den2 = 0, 0, 0
    f_avg, g_avg = 0, 0
    for i in range(len(f)):
        f_avg += f[i]
        g_avg += g[i]
    f_avg = f_avg / len(f)
    g_avg = g_avg / len(g)
    for i in range(len(f)):
        num += (f[i] - f_avg) * (g[i] - g_avg)
        den1 += (f[i] - f_avg)**2
        den2 += (g[i] - g_avg)**2
    return num / ((den1 * den2)**0.5)


# In[ ]:


labels = ['pollution', 'dew', 'temp', 'pressure', 'wind_dir', 'wind_spd', 'snow', 'rain']
f = train_data[:, 0]
for i in range(8):
    g = train_data[:, i]
    print(labels[i], ' '*(20-len(labels[i])), corr(f, g))


# In[ ]:


selected_train_data, selected_test_data = [x[[0, 1, 5]] for x in train_data], [x[[0, 1, 5]] for x in test_data]

x_train_per_day, y_train_per_day = create_dataset(selected_train_data, 11, 12)
x_test_per_day, y_test_per_day = create_dataset(selected_test_data, 11, 12)
print(x_train_per_day.shape, y_train_per_day.shape)
print(x_test_per_day.shape, y_test_per_day.shape)


# ## Training Models

# ### Simple RNN

# In[ ]:


model = Sequential([
  SimpleRNN(100, input_shape=(11, 3)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_per_day = model.fit(x_train_per_day, y_train_per_day, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_per_day.model.predict(x_test_per_day)

mse = mean_squared_error(y_test_per_day, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_per_day)
show_train_loss_history(trained_per_day, "Training Loss", False)


# ### LSTM

# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(11, 3)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_per_day = model.fit(x_train_per_day, y_train_per_day, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_per_day.model.predict(x_test_per_day)

mse = mean_squared_error(y_test_per_day, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_per_day)
show_train_loss_history(trained_per_day, "Training Loss", False)


# ### GRU

# In[ ]:


model = Sequential([
  GRU(100, input_shape=(11, 3)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_per_day = model.fit(x_train_per_day, y_train_per_day, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_per_day.model.predict(x_test_per_day)

mse = mean_squared_error(y_test_per_day, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test_per_day)
show_train_loss_history(trained_per_day, "Training Loss", False)


# # Missing Values

# ## Handling Missing Values

# In[ ]:


num_missing_data = int(0.2 * len(train_data))
missing_train_data = train_data.copy()

for i in range(8):
  random_indexes = random.sample(range(0, len(missing_train_data)), num_missing_data)
  missing_train_data[random_indexes, i] = np.NaN


# In[ ]:


pd.DataFrame(missing_train_data).describe()


# In[ ]:


mean_cols = []
for col in range(8):
  mean_cols.append(np.nanmean(missing_train_data[:,col]))

def mean_imputation(data, col):
  data = data.copy()
  for c in range(8):
    if c==col:
      continue
    nan_index = np.isnan([x[c] for x in data])
    data[nan_index, c] = mean_cols[c]
  return data

def impute_col_elems(data, col):
  data = np.array(data)
  nan_index = np.isnan(data)
  data[nan_index] = mean_cols[col]
  return list(data)


# In[ ]:


def give_index_smallest_elements(d_list, k):
  res = sorted(range(len(d_list)), key = lambda sub: d_list[sub])[:k] 
  return res


# In[ ]:


def knn_imputation(data, k):
  data = data.copy()
  for col in range(8):
    print(col)
    data_i = mean_imputation(data, col)
    nan_index = np.isnan([x[col] for x in data])
    for i in range(len(data)):
      if nan_index[i]:
        distances = [np.linalg.norm(v[np.arange(8)!=col]-data_i[i][np.arange(8)!=col]) for v in data_i]
        min_indexes = give_index_smallest_elements(distances, k)
        neigh_values = [data_i[index][col] for index in min_indexes]
        neigh_values = impute_col_elems(neigh_values, col)
        data[i][col] = np.mean(neigh_values)
  return data


# In[ ]:


x_trained_filled = knn_imputation(missing_train_data[:], 540)

mse_list = []
for col in range(8):
  mse = mean_squared_error(x_trained_filled[:,col], train_data[:,col])
  mse_list.append(mse)
  print('   MSE for column', col, ' : ', mse) 
print(' mean : ', np.mean(mse_list))


# In[ ]:


# Median
imputer = SimpleImputer(strategy="median")
imputer.fit(missing_train_data)
x_trained_filled_median = imputer.transform(missing_train_data)

# Mean
imputer = SimpleImputer(strategy="mean")
imputer.fit(missing_train_data)
x_trained_filled_mean = imputer.transform(missing_train_data)

# Most Frequent
imputer = SimpleImputer(strategy="most_frequent")
imputer.fit(missing_train_data)
x_trained_filled_most_frequent = imputer.transform(missing_train_data)

# KNN
imputer = KNNImputer(n_neighbors=540)
imputer.fit(missing_train_data)
x_train_filled_knn = imputer.transform(missing_train_data)

# Iterative
imputer = IterativeImputer(random_state=0, estimator=ExtraTreesRegressor())
imputer.fit(missing_train_data)
x_train_filled_iterative = imputer.transform(missing_train_data)


# In[ ]:


filled_data_labels = ['Median', 'Mean', 'Most Frequent', 'KNN', 'Iterative']
filled_data = [x_trained_filled_median, x_trained_filled_mean, x_trained_filled_most_frequent, x_train_filled_knn, x_train_filled_iterative]

for i in range(len(filled_data)):
  print(filled_data_labels[i])
  data = filled_data[i]
  mse_list = []
  for col in range(8):
    mse = mean_squared_error(data[:,col], train_data[:,col])
    mse_list.append(mse)
    print('   MSE for column', col, ' : ', mse) 
  print(' mean : ', np.mean(mse_list))
  print()


# ## Training Model

# In[ ]:


x_train, y_train = create_dataset(x_trained_filled, 11)
x_test, y_test = create_dataset(test_data, 11)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ### LSTM

# In[ ]:


model = Sequential([
  LSTM(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_filled_lstm = model.fit(x_train, y_train, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_filled_lstm.model.predict(x_test)

mse = mean_squared_error(y_test, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test)
show_train_loss_history(trained_filled_lstm, "Training Loss", False)


# ### GRU

# In[ ]:


model = Sequential([
  GRU(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_filled_gru = model.fit(x_train, y_train, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_filled_gru.model.predict(x_test)

mse = mean_squared_error(y_test, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test)
show_train_loss_history(trained_filled_gru, "Training Loss", False)


# # Simple Models with 100 Epochs

# ## Simple RNN

# In[11]:


model = Sequential([
    SimpleRNN(100, input_shape=(11, 8)),
    Dense(1, activation='tanh')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_rnn = model.fit(x_train, y_train, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_rnn.model.predict(x_test)
wanted = [predicted[i] for i in range(0, len(predicted), 12)]
actual = [y_test[i] for i in range(0, len(predicted), 12)]
mse = mean_squared_error(actual, wanted) 
print("MSE = ", mse) 
plot_prediction(wanted, actual)
show_train_loss_history(trained_rnn, "Training Loss Plot", False)


# ## LSTM 

# In[12]:


model = Sequential([
  LSTM(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_filled_lstm = model.fit(x_train, y_train, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_filled_lstm.model.predict(x_test)

mse = mean_squared_error(y_test, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test)
show_train_loss_history(trained_filled_lstm, "Training Loss", False)


# ## GRU

# In[13]:


model = Sequential([
  GRU(100, input_shape=(11, 8)),
  Dense(1, activation='tanh')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

start=time()
trained_filled_gru = model.fit(x_train, y_train, epochs=100, shuffle=True, batch_size=10)
train_time = time()-start
print("Training time:", train_time)

predicted = trained_filled_gru.model.predict(x_test)

mse = mean_squared_error(y_test, predicted) 
print("MSE = ", mse)
plot_prediction(predicted, y_test)
show_train_loss_history(trained_filled_gru, "Training Loss", False)

