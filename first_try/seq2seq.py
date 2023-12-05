# Implementation of time series forecasting using GAF image translation technique into seq2seq model

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

## 1. Initial data preparation #########################################################################

# Import time series
df = pd.read_hdf('src/RICO1_Dataset.hdf', key='all')
df['_time'] = pd.to_datetime(df['_time'])
df.set_index('_time', inplace=True)
# Resampling in 10 min intervals
df = df.resample('10T').mean()
rtd1 = df['B.RTD1']

# Split data into train and test
train_size = int(len(rtd1) * 0.8)
test_size = len(rtd1) - train_size
train, test = rtd1[0:train_size], rtd1[train_size:len(rtd1)]

# Scaling datasets
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Transforming into GAF
gaf = GramianAngularField(image_size=175, method='summation')
train_gaf = gaf.fit_transform(train_scaled.reshape(1, -1))
test_gaf = gaf.transform(test_scaled.reshape(1, -1))

# Plot GAF Images
fig, axs = plt.subplots(1, 2, figsize=(12,6))
axs[0].imshow(train_gaf[0], cmap='rainbow', origin='lower')
axs[0].set_title('Train GAF Image')
axs[1].imshow(test_gaf[0], cmap='rainbow', origin='lower')
axs[1].set_title('Train GAF Image')
plt.colorbar()
plt.show()

## 2. Model development #########################################################################

input_shape = (None, train_gaf.shape[1], train_gaf.shape[2])  # Adapt to your GAF image size
latent_dim = 256 # Latent dimensionality of the encoding space.

# Encoder
encoder_inputs = Input(shape=input_shape)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=input_shape)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(train_gaf.shape[1] * train_gaf.shape[2], activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

## 3. Training #########################################################################

cp = ModelCheckpoint('model/', save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=100, callbacks=[cp,es])
