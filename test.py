import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import tensorflow as tf

import keras, sys, time, os, warnings, cv2

from keras.models import *
from keras.layers import *

import numpy as np
import pandas as pd

import glob
import os

X_train, y_train, y_train0 = [], [], []

import pickle as pkl

with open("./train_data.pkl", "rb") as f:
    df = pkl.load(f)

    df = df.replace(to_replace='None', value=np.nan).dropna()

    cols = df.columns[:-1]
    for col in cols:
        df[col] = df[col].astype(np.float32)
    y_train, y_train0, nm_landmarks = get_y_as_heatmap(df, 96, 96, 5)

    X_train = df['Image'].values.tolist()

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train0 = np.array(y_train0)

print(X_train.shape, y_train.shape, y_train0.shape)
# print(X_test.shape,y_test)


prop_train = 0.9
Ntrain = int(X_train.shape[0] * prop_train)
X_tra, y_tra, X_val, y_val = X_train[:Ntrain], y_train[:Ntrain], X_train[Ntrain:], y_train[Ntrain:]
del X_train, y_train

input_height, input_width = 96, 96
## output shape is the same as input
output_height, output_width = input_height, input_width

# nClasses = 15
nClasses = 9  # ironman

model = []

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("./model.h5")
model = loaded_model
model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")

print(model.output.shape)

y_pred = model.predict(X_val)
y_pred = y_pred.reshape(-1, output_height, output_width, nClasses)

Nlandmark = y_pred.shape[-1]
for i in range(10):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(X_val[i, :, :], cmap="gray")
    ax.axis("off")

    fig = plt.figure(figsize=(20, 3))
    count = 1
    for j in range(9):
        ax = fig.add_subplot(2, Nlandmark, count)
        # 		ax.set_title(lab[:10] + "\n" + lab[10:-2])
        ax.axis("off")
        count += 1
        ax.imshow(y_pred[i, :, :, j])
        if j == 0:
            ax.set_ylabel("prediction")

    for j in range(9):
        ax = fig.add_subplot(2, Nlandmark, count)
        count += 1
        ax.imshow(y_val[i, :, :, j])
        ax.axis("off")
        if j == 0:
            ax.set_ylabel("true")

    # img_name = './img_{:03d}.png'.format(i)
    # plt.savefig( img_name, bbox_inches='tight', dpi=(300) )
    plt.show()
    plt.close()