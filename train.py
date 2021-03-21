
import cv2
import time
import sys
import pandas as pd
import keras
import pickle as pkl
import os
import glob
import numpy as np
from keras.layers import *
from keras.models import *
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  # (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  # (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


def generate_hm(height, width, landmarks, s=3):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):

            hm[:, :, i] = gaussian_k(landmarks[i][0],
                                     landmarks[i][1],
                                     s, height, width)
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm


def get_y_as_heatmap(df, height, width, sigma):

    columns_lmxy = df.columns[:-1]  # the last column contains Image
    columns_lm = []
    for c in columns_lmxy:
        c = c[:-2]
        if c not in columns_lm:
            columns_lm.extend([c])

    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "_x"].iloc[i]
            y = df[colnm + "_y"].iloc[i]
            if pd.isnull(x) or pd.isnull(y):
                x, y = -1, -1
            landmarks.append([x, y])

        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)
    return(y_train, df[columns_lmxy], columns_lmxy)


def load2d(test=False, width=256, height=256, sigma=5):
    if test:
        path = "./test_data.pkl"
    else:
        path = "./train_data.pkl"

    df = pd.read_pickle(path)
    df = df.replace(to_replace='None', value=np.nan).dropna()

    cols = df.columns[:-1]
    for col in cols:
        df[col] = df[col].astype(np.float32)

    y, y0, nm_landmarks = get_y_as_heatmap(
        df, height, width, sigma)

    X = df['Image'].values

    return X, y, y0, nm_landmarks


input_height, input_width, sigma = 256, 256, 5


# output shape is the same as input
output_height, output_width = input_height, input_width

X_train, y_train, y_train0, nm_landmarks = load2d(test=False, sigma=sigma)
X_test,  y_test, _, _ = load2d(test=True, sigma=sigma)

print(X_train.shape, y_train.shape, y_train0.shape)
print(X_test.shape, y_test.shape)

Nplot = y_train.shape[3]+1

for i in range(3):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(2, Nplot/2, 1)
    ax.imshow(X_train[i, :, :, 0], cmap="gray")
    ax.set_title("input")
    for j, lab in enumerate(nm_landmarks[::2]):
        ax = fig.add_subplot(2, Nplot/2, j+2)
        ax.imshow(y_train[i, :, :, j], cmap="gray")
        ax.set_title(str(j) + "\n" + lab[:-2])
    plt.show()

# nClasses = 15
nClasses = y_train.shape[3]  # 9 keypoints


# model = []

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model

# loaded_model.load_weights("my_model_weights.hdf5")
# model = loaded_model
# model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")

# print(model.output.shape)
