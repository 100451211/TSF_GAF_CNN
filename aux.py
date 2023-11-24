import numpy as np
import matplotlib.pyplot as plt
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