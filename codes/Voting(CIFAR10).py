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

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chunk_size = len(X_train) // size
# val_chunk_size = len(X_test) // size

print(f"Number of nodes: {size} || the chunk size to train each node: {chunk_size}")

local_X_train = X_train[rank * chunk_size: (rank + 1) * chunk_size]
local_y_train = y_cat_train[rank * chunk_size: (rank + 1) * chunk_size]

# local_X_val = X_test[rank * val_chunk_size: (rank + 1) * val_chunk_size]
# local_y_val = y_cat_test[rank * val_chunk_size: (rank + 1) * val_chunk_size]


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

local_model = build_model()
local_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# local_model.summary(line_length = 100)
batch_size = 32
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(local_X_train, local_y_train, batch_size=batch_size)
steps_per_epoch = len(local_X_train) // batch_size

start_time = time.time()
r = local_model.fit(train_generator,
                    epochs=50,
                    steps_per_epoch=steps_per_epoch,
                    # validation_data=(local_X_val, local_y_val),
                    verbose=2)
end_time = time.time()
training_time = end_time - start_time

print(f"Node {rank} Training time:", training_time)

evaluation = local_model.evaluate(local_X_train, local_y_train,verbose=0)

print(f"Node {rank} Training Accuracy:", evaluation)

all_predictions = comm.gather(local_model.predict(X_test,verbose=0), root=0)
all_evaluations = comm.gather(evaluation[0], root=0)

comm.Barrier()

if rank == 0:
    def majority_vote(predictions):
        sum_predictions = np.sum(predictions, axis=0)
        return np.argmax(sum_predictions)
    start_time = time.time()
    global_predictions = [majority_vote(predictions) for predictions in zip(*all_predictions)]
    end_time = time.time()
    majority_time = end_time - start_time
    print(f"Node {rank} Time to find majority: ", majority_time)
    global_accuracy = np.mean(global_predictions == np.argmax(y_cat_test, axis=1))
    print("Global Model Accuracy on test data:", global_accuracy)
