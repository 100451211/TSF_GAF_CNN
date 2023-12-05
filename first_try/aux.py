import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor # pip3 install scikeras
from sklearn.model_selection import GridSearchCV
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
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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

def tuning(X_train_gaf, y_train_cv):
    # Define the model
    def create_model(num_filters=[32], kernel_size=[(3,3)], pool_size=[(2,2)], dense_units=[64], learning_rate=0.001):
        model = Sequential()
        # Add each convolutional layer.
        for i, (filters, kernel) in enumerate(zip(num_filters, kernel_size)):
            # For the first layer, specify the input shape.
            if i == 0:
                model.add(Conv2D(filters, kernel, activation='relu', input_shape=(image_size, image_size, 1)))
            else:
                model.add(Conv2D(filters, kernel, activation='relu'))
            model.add(MaxPooling2D(pool_size=pool_size[i]))
        model.add(Flatten())
        # Add dense layers.
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1)) # Change this if you have more than one output neuron
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
        return model

    # Wrap the model with KerasRegressor
    model = KerasRegressor(build_fn=create_model, verbose=1)

    # Define the parameter grid to search
    param_grid = {
        'num_filters': [[32], [64]],
        'kernel_size': [[(3, 3)], [(5, 5)]],
        'pool_size': [[(2, 2)], [(3, 3)]],
        'dense_units': [[64], [128]],
        'batch_size': [32, 64],
        'epochs': [10, 20],
        'learning_rate': [0.001, 0.0001]
    }

    # Create a GridSearchCV instance
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    print("Starting grid search...")
    grid_result = grid.fit(X_train_gaf, y_train_cv)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.cv_results_['params'], grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score']:
        print("%f (%f) with: %r" % (mean_score, scores.std(), params))
    