import pyts
import math
from aux import *
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from pyts.image import GramianAngularField
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# 1. Read dataset 
df = pd.read_hdf('../RICO1_Dataset.hdf', key='all')
# Displaying the first few rows of the dataset
# print(data.head())


# 2. Resampling in 10 minute interval
df['_time'] = pd.to_datetime(df['_time'])
df.set_index('_time', inplace=True)
df = df.resample('10T').mean()
rtd1 = df['B.RTD1'] #2449 rows - representing values each 10 min 
# print(rtd1.head())


# 3. Splitting the dataset into train and test sets (80:20)
train_size = int(len(rtd1) * 0.8)
test_size = len(rtd1) - train_size
train, test = rtd1[0:train_size], rtd1[train_size:len(rtd1)]
# print(f'train shape: {train.shape}, \n train: {train}, \n test shape: {test.shape}, \n test: {test}')


# 4. Data Preprocessing - Normalizing the train data
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train.values.reshape(-1, 1))


# 5. Create the dataset with look back and forecast horizon=36
look_back = 300
forecast_horizon = 36
trainX, trainY = create_dataset(train, look_back, forecast_horizon)
testX, testY = create_dataset(test, look_back, forecast_horizon)
print(f'trainX shape: {trainX.shape}, \n trainY shape: {trainY.shape}, \n testX shape: {testX.shape}, \n testY shape: {testY.shape}')

# 6. Reshaping the train and test sets into 2D appropiate for GAF transformation
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
testX = testX.reshape(testX.shape[0], testX.shape[1])
trainY = trainY.reshape(trainY.shape[0], trainY.shape[1])
testY = testY.reshape(testY.shape[0], testY.shape[1])

# 7. Setting up GAF transformation
image_size = 300 
gasf = GramianAngularField(image_size=image_size, method='summation')

# 8. Transforming the data & plotting
print(f'trainX shape: {trainX.shape}')
X_gasf_train = gasf.fit_transform(trainX)
X_gasf_train = X_gasf_train.reshape(X_gasf_train.shape[0], image_size, image_size, 1)  # Reshape for CNN
#print(f'X_gasf_train shape: {X_gasf_train.shape}\n X_gasf_train[0]:{X_gasf_train[0]}')
# plt.imshow(X_gasf_train[0], cmap='gray', origin='lower')
# plt.title('GASF Train', fontsize=16)
# plt.colorbar()
# plt.show()

X_gasf_test = gasf.transform(testX)
X_gasf_test = X_gasf_test.reshape(X_gasf_test.shape[0], image_size, image_size, 1)  # Reshape for CNN
print(f'X_gasf_train shape: {X_gasf_test.shape}\n X_gasf_train:{X_gasf_test}')
# plt.imshow(X_gasf_test[0], cmap='gray', origin='lower')
# plt.title('GASF Test ', fontsize=16)
# plt.colorbar()
# plt.show()


# 8. Build the CNN model
# 8.1. CNN model for GASF
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(forecast_horizon)  # Output layer
])

cp = ModelCheckpoint('model/', save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mse'])

# 9. Fit the CNN models
train_hist = model.fit(X_gasf_train, trainY, epochs=15, batch_size=64, validation_split=0.2 , callbacks=[cp,es])
# print(history)
#plot_loss(train_hist)


# 10. Make predictions
trainPredict = model.predict(X_gasf_train)
#print(f'trainPredict shape: {trainPredict.shape}\n trainPredict:{trainPredict}')


# 11. Invert predictions

print(f'trainPredict shape: {trainPredict.shape}\n trainY.shape:{trainY.shape}')
trainPredict = scaler.inverse_transform(trainPredict).flatten()
#trainY = scaler.inverse_transform(trainY).flatten()
trainY = trainY.flatten()


# 12. Plot train_results line plot
start_point = look_back + forecast_horizon - 1
time_index = pd.date_range(start=df.index[start_point], periods=len(trainPredict), freq='10T')

# Create a DataFrame for plotting
comparison_df = pd.DataFrame({
    'Actuals': trainY,
    'Predictions': trainPredict
}, index=time_index)

# Plot the actuals vs predictions
plt.figure(figsize=(15, 7))
plt.plot(comparison_df.index, comparison_df['Actuals'], label='Actual Values')
plt.plot(comparison_df.index, comparison_df['Predictions'], label='Predictions')
plt.title('Comparison of Actuals and Predictions')
plt.xlabel('Time')
plt.ylabel('B.RTD1 Value')
plt.legend()
plt.show()


# 13. Evaluate the model
mse = model.evaluate(X_gasf_test, testY, verbose=0)
print('mse[0]: %.2f \n mse[1]: %.2f' % (mse[0], mse[1]))



# ################################################################################################################################

# # Hyperparameter tuning for model 


