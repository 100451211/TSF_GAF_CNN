import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout


def plot_loss(hist):
  plt.plot(hist.history['loss'], label='Training Loss')
  plt.plot(hist.history['val_loss'], label='Validation Loss')
  plt.title(f'Model -  Loss plot')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='best')
  plt.show()
  plt.close()

# # Previous version
# def create_dataset(dataset, look_back, forecast_horizon):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - forecast_horizon + 1):
#         a = dataset[i:(i + look_back)]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back:i + look_back + forecast_horizon])
#     return np.array(dataX), np.array(dataY)

# Function to create dataset with sliding windows
def create_dataset(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[(i + look_back):(i + look_back + forecast_horizon), 0])
    return np.array(X), np.array(y)


def build_model(input_shape, forecast_horizon, learning_rate):
    print(f'build.model: input_shape = {input_shape}')
    #input_shape = (300, 300, 1) # (batch_size, height, width, channels)
    model = Sequential([
        # Start with a 1st Convolutional layer
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(300, 300, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        
        # Conv2D(64, (3, 3), activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling2D((2, 2)),
        
        # Flatten and add a dense layer
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # Output layer with 'forecast_horizon' units
        Dense(forecast_horizon)
    ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model