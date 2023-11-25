import pyts
import math
from aux import *
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
from keras.optimizers.legacy import Adam
from pyts.image import GramianAngularField
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,load_model
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# 1. Read dataset 
df = pd.read_hdf('src/RICO1_Dataset.hdf', key='all')
# Displaying the first few rows of the dataset
# print(data.head())


# 2. Resampling in 10 minute interval
df['_time'] = pd.to_datetime(df['_time'])
df.set_index('_time', inplace=True)
df = df.resample('10T').mean()
rtd1 = df['B.RTD1'] #2449 rows - representing values each 10 min 
# print(rtd1.head())

# 2.2 Apply Moving Average for Noise Reduction
window_size = 3  # Choose a window size that suits your data
rtd1_smoothed = rtd1.rolling(window=window_size, center=True).mean().dropna()


# 3. Splitting the dataset into train and test sets (80:20)
train_size = int(len(rtd1_smoothed) * 0.8)
test_size = len(rtd1_smoothed) - train_size
train, test = rtd1_smoothed[0:train_size], rtd1_smoothed[train_size:len(rtd1_smoothed)]


# 4. Data Preprocessing - Normalizing the train data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

look_back = 175
forecast_horizon = 36

image_size = 175
gasf = GramianAngularField(image_size=image_size, method='summation')

# Define TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Store validation results
val_mse_scores = []

# Store predictions and actuals
all_predictions = []
all_actuals = []

for train_index, val_index in tscv.split(train_scaled):
    print(f"Training on {len(train_index)} samples, validating on {len(val_index)} samples.")
    #print(f"Indices for train: {train_index}, val: {val_index}")

    # Prepare training and validation sets
    X_train_cv, y_train_cv = create_dataset(train_scaled[train_index], look_back, forecast_horizon)
    X_val_cv, y_val_cv = create_dataset(train_scaled[val_index], look_back, forecast_horizon)

    print(f"Shapes - X_train: {X_train_cv.shape}, y_train: {y_train_cv.shape}, X_val: {X_val_cv.shape}, y_val: {y_val_cv.shape}")

    if X_train_cv.size == 0 or X_val_cv.size == 0:
        print("Found an empty array, skipping this split.")
        continue

    # Transform data into GAF images
    X_train_gaf = gasf.fit_transform(X_train_cv.reshape(X_train_cv.shape[0], look_back))
    X_val_gaf = gasf.transform(X_val_cv.reshape(X_val_cv.shape[0], look_back))

    if X_train_cv.shape[0] == 0 or X_val_cv.shape[0] == 0:
        print("Not enough data for this fold, skipping...")
        continue
    
    # Reshape for CNN
    X_train_gaf = X_train_gaf.reshape(X_train_gaf.shape[0], image_size, image_size, 1)
    X_val_gaf = X_val_gaf.reshape(X_val_gaf.shape[0], image_size, image_size, 1)

    # Define callbacks
    cp = ModelCheckpoint('model/', save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Build and train the model
    model = build_model((image_size, image_size, 1), forecast_horizon, learning_rate=0.0001)
    X_train_resh = X_train_gaf.reshape(-1, image_size, image_size, 1)
    X_val_resh = X_val_gaf.reshape(-1, image_size, image_size, 1)
    history = model.fit(X_train_gaf, y_train_cv, epochs=70, batch_size=64, callbacks=[cp, es], validation_data=(X_val_gaf, y_val_cv))

    # Predict on validation set
    val_predictions = model.predict(X_val_gaf)
    val_predictions_inv = scaler.inverse_transform(val_predictions)

    # Inverse transform actual validation data
    y_val_inv = scaler.inverse_transform(y_val_cv)

    # Store predictions and actuals
    all_predictions.append(val_predictions_inv)
    all_actuals.append(y_val_inv)

    # Evaluate on the validation set
    val_mse = model.evaluate(X_val_gaf, y_val_cv, verbose=0)[1]
    val_mse_scores.append(val_mse)

# Average validation MSE
average_val_mse = sum(val_mse_scores) / len(val_mse_scores)
print(f'Average Validation MSE: {average_val_mse}')

# Optional: Plotting the training and validation loss
# Assuming 'history' contains the training history of the last fold
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Predictions
num_plots = len(all_predictions)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))

for i, (preds, actuals) in enumerate(zip(all_predictions, all_actuals)):
    # Plot on the ith subplot
    axes[i].plot(preds[:, 0], label='Predictions')  # Plotting first forecasted value
    axes[i].plot(actuals[:, 0], label='Actual')
    axes[i].set_title(f'Fold {i+1} Predictions vs Actuals')
    axes[i].set_xlabel('Time Steps')
    axes[i].set_ylabel('B.RTD1 Value')
    axes[i].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
time = datetime.now().strftime("%d-%H%M")
plt.savefig('/images/predictions_vs_actuals{time}.png')

plt.show()
