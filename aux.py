import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn import TimeSeriesSplit

def plot_loss(hist):
  plt.plot(hist.history['loss'], label='Training Loss')
  plt.plot(hist.history['val_loss'], label='Validation Loss')
  plt.title(f'Model -  Loss plot')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='best')
  plt.show()
  plt.close()

# Function to create dataset with sliding windows
def create_dataset(dataset, look_back, forecast_horizon):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back + forecast_horizon])
    return np.array(dataX), np.array(dataY)


def find_optimal_look_back(df, forecast_horizon, look_back_values, n_splits=5):
    # Split the dataset into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    # Prepare the full dataset for cross-validation
    dataset = np.concatenate((train_scaled, test_scaled))

    # Initialize time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Record the performance for each look_back value
    look_back_performance = {}

    for look_back in look_back_values:
        mse_scores = []

        for train_index, test_index in tscv.split(dataset):
            # Split data using the indices provided by the cross-validator
            train, test = dataset[train_index], dataset[test_index]

            # Prepare the dataset with the current look_back
            trainX, trainY = create_dataset(train, look_back, forecast_horizon)
            testX, testY = create_dataset(test, look_back, forecast_horizon)

            # Train and make predictions
            model = train_model(trainX, trainY)
            predictions = model.predict(testX)

            # Evaluate predictions
            mse = mean_squared_error(testY, predictions)
            mse_scores.append(mse)

        # Store the mean MSE for the current look_back
        look_back_performance[look_back] = np.mean(mse_scores)

    # Select the best look_back value
    best_look_back = min(look_back_performance, key=look_back_performance.get)
    return best_look_back, look_back_performance