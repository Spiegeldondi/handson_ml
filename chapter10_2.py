#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:38:12 2019

@author: domi
"""

"""
Building an Image Classifier Using the Sequential API
"""

# %% de-comment and run for CPU only usage (confirmed by manually checking
# GPU memory-usage via "nvidia-smi" in the terminal: ~ 10% vs. ~ 90% usage)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% import packages

import numpy as np
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# create validation set since there are only a train and test set
# scale pixel idntensities to 0-1 range since we are using Gradient Descent

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %% create the model using sequential API

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# compile the model

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %% train and evaluate the model

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# %% plot learning curves

import pandas as pd

import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show


















