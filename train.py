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


def load2d(test=False, width=96, height=96, sigma=5):
    if test:
        path = "./test_data.pkl"
    else:
        path = "./train_data.pkl"

    df = pd.read_pickle(path)

    # if test:
    #     path = "./test_data_aug.pkl"
    #     df.append(pd.read_pickle(path))
    #     path = "./train_data_aug.pkl"
    #     df.append(pd.read_pickle(path))

    df = df.replace(to_replace='None', value=np.nan).dropna()

    cols = df.columns[:-1]
    for col in cols:
        df[col] = df[col].astype(np.float32)

    y, y0, nm_landmarks = get_y_as_heatmap(
        df, height, width, sigma)

    X = df['Image'].values.tolist()
    X = np.array(X)  # fix for weird shape (959, ) to (959, height, width)
    X = np.expand_dims(X, axis=3)

    return X, y, y0, nm_landmarks


input_height, input_width, sigma = 96, 96, 5


# output shape is the same as input
output_height, output_width = input_height, input_width

X_train, y_train, y_train0, nm_landmarks = load2d(test=False, sigma=sigma)
# X_test,  y_test, _, _ = load2d(test=True, sigma=sigma)

print(X_train.shape, y_train.shape, y_train0.shape)
# print(X_test.shape, y_test.shape)

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
n = 32*5
nfmp_block1 = 64
nfmp_block2 = 128
IMAGE_ORDERING = "channels_last"

IMAGE_ORDERING = "channels_last"
img_input = Input(shape=(input_height, input_width, 1))

# Encoder Block 1
x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
           name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same',
           name='block1_conv2', data_format=IMAGE_ORDERING)(x)
block1 = MaxPooling2D((2, 2), strides=(
    2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

# Encoder Block 2
x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
           name='block2_conv1', data_format=IMAGE_ORDERING)(block1)
x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same',
           name='block2_conv2', data_format=IMAGE_ORDERING)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                 data_format=IMAGE_ORDERING)(x)

# bottoleneck
o = (Conv2D(n, (int(input_height/4), int(input_width/4)),
            activation='relu', padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
o = (Conv2D(n, (1, 1), activation='relu', padding='same',
            name="bottleneck_2", data_format=IMAGE_ORDERING))(o)


# upsamping to bring the feature map size to be the same as the one from block1
o_block1 = Conv2DTranspose(nfmp_block1, kernel_size=(2, 2),  strides=(
    2, 2), use_bias=False, name='upsample_1', data_format=IMAGE_ORDERING)(o)
o = Add()([o_block1, block1])
output = Conv2DTranspose(nClasses,    kernel_size=(2, 2),  strides=(
    2, 2), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING)(o)


# Decoder Block
output = Reshape((output_width*input_height*nClasses, 1))(output)
model = Model(img_input, output)
model.summary()
model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")


def find_weight(y_tra):
    weight = np.zeros_like(y_tra)
    count0, count1 = 0, 0
    for irow in range(y_tra.shape[0]):
        for ifeat in range(y_tra.shape[-1]):
            if np.all(y_tra[irow,:,:,ifeat] == 0):
                value = 0
                count0 += 1
            else:
                value = 1
                count1 += 1
            weight[irow,:,:,ifeat] = value
    print("N landmarks={:5.0f}, N missing landmarks={:5.0f}, weight.shape={}".format(
        count0,count1,weight.shape))
    return(weight)

def flatten_except_1dim(weight, ndim=2):
    '''
    change the dimension from:
    (a,b,c,d,..) to (a, b*c*d*..) if ndim = 2
    (a,b,c,d,..) to (a, b*c*d*..,1) if ndim = 3
    '''
    n = weight.shape[0]
    if ndim == 2:
        shape = (n, -1)
    elif ndim == 3:
        shape = (n, -1, 1)
    else:
        print("Not implemented!")
    weight = weight.reshape(*shape)
    return(weight)


prop_train = 0.9
Ntrain = int(X_train.shape[0]*prop_train)
X_tra, y_tra, X_val, y_val = X_train[:Ntrain], y_train[:Ntrain], X_train[Ntrain:], y_train[Ntrain:]
del X_train, y_train

weight_val = find_weight(y_val)
weight_val = flatten_except_1dim(weight_val)
y_val_fla = flatten_except_1dim(y_val, ndim=3)

# print("weight_tra.shape={}".format(weight_tra.shape))
print("y_val_fla.shape={}".format(y_val_fla.shape))
print(model.output.shape)

nb_epochs = 300
batch_size = 8
const = 10
history = {"loss": [], "val_loss": []}
for iepoch in range(nb_epochs):
    start = time.time()

    x_batch, y_batch = X_tra, y_tra
    y_batch_fla = flatten_except_1dim(y_batch, ndim=3)

    hist = model.fit(x_batch,
                     y_batch_fla*const,
                     validation_data=(X_val, y_val_fla*const,weight_val),
                     batch_size=2,
                     epochs=1,
                     verbose=1)
    history["loss"].append(hist.history["loss"][0])
    history["val_loss"].append(hist.history["val_loss"][0])
    end = time.time()
    print("Epoch {:03}: loss {:6.4f} val_loss {:6.4f} {:4.1f}sec".format(
        iepoch+1, history["loss"][-1], history["val_loss"][-1], end-start))

for label in ["val_loss", "loss"]:
    plt.plot(history[label], label=label)
plt.legend()
plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

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
