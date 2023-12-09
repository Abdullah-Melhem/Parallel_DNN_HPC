import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
from tensorflow import keras
from tensorflow.keras import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mpi4py import MPI
import numpy as np
from mpi4py import MPI
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# Function to build a CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Federated Averaging implementation
def federated_averaging(comm, X_train, y_train, num_iterations=10):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize a global CNN model
    global_model = build_model()

    for iteration in range(num_iterations):
        # Distribute the global model to all nodes
        global_weights_flat = np.empty_like(np.concatenate([w.flatten() for w in global_model.get_weights()]))
        if rank == 0:
            global_weights_flat = np.concatenate([w.flatten() for w in global_model.get_weights()])

        # Broadcast the global weights to all nodes
        comm.Bcast([global_weights_flat, MPI.FLOAT], root=0)

        # Split the received weights into the original shapes
        shapes = [w.shape for w in global_model.get_weights()]
        global_weights = [global_weights_flat[offset:offset+np.prod(shape)].reshape(shape) for offset, shape in zip([0]+np.cumsum([np.prod(shape) for shape in shapes])[:-1].tolist(), shapes)]

        # Train local CNN models on each node
        local_model = build_model()
        local_model.set_weights(global_weights)

        # Train local model on local data
        local_model.fit(X_train, y_train, epochs=1, verbose=0)

        # Get the model weights from each node
        local_weights = local_model.get_weights()

        # Flatten and concatenate local weights
        local_weights_flat = np.concatenate([w.flatten() for w in local_weights])

        # Aggregate model weights using averaging
        comm.Allreduce(MPI.IN_PLACE, [local_weights_flat, MPI.FLOAT], op=MPI.SUM)

        # Average the weights
        local_weights_flat /= size

        # Split the received weights into the original shapes
        local_weights = [local_weights_flat[offset:offset+np.prod(shape)].reshape(shape) for offset, shape in zip([0]+np.cumsum([np.prod(shape) for shape in shapes])[:-1].tolist(), shapes)]

        # Set the updated weights to the global model
        global_model.set_weights(local_weights)

    return global_model

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Split the data among MPI ranks
chunk_size = len(X_train) // comm.Get_size()
local_X_train = X_train[rank * chunk_size: (rank + 1) * chunk_size]
local_y_train = y_cat_train[rank * chunk_size: (rank + 1) * chunk_size]

# Call the Federated Averaging function
start_time = time.time()
global_model = federated_averaging(comm, local_X_train, local_y_train, num_iterations=10)
end_time = time.time()
training_time = end_time - start_time
print(f"Node {rank} Time to find training_time: ", training_time)
# Use the final global model for prediction or evaluation
if rank == 0:
    evaluation = global_model.evaluate(X_test, y_cat_test,verbose=0)
    print(f'The Global Model Performace :"{evaluation}')
