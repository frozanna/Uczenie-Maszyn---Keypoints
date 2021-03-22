from sklearn.utils import shuffle
import os
import pandas as pd
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
import matplotlib.pyplot as plt

DATA_PATH = r'./Dataset'


def load():
    df = pd.DataFrame(columns=['Keypoint1_x', 'Keypoint1_y', 'Keypoint2_x', 'Keypoint2_y',
                               'Keypoint3_x', 'Keypoint3_y', 'Keypoint4_x', 'Keypoint4_y',
                               'Keypoint5_x', 'Keypoint5_y', 'Keypoint6_x', 'Keypoint6_y',
                               'Keypoint7_x', 'Keypoint7_y', 'Keypoint8_x', 'Keypoint8_y',
                               'Keypoint9_x', 'Keypoint9_y', 'Image'])

    subdirs = [DATA_PATH + '/' + name for name in os.listdir(
        DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))]

    for subdir in subdirs:
        keypoint_file = glob(subdir + '/*keypoints')
        file = open(keypoint_file[0], "r")
        file_content = file.read()
        arr = [line.split() for line in file_content.split('\n')]

        color_subdir = os.path.join(subdir, 'color')
        for i, filename in enumerate(os.listdir(color_subdir)):
            image = np.array(cv2.imread(os.path.join(
                color_subdir, filename), cv2.IMREAD_GRAYSCALE))
            image = image / 255
            arr[i].append(image)

        array_df = pd.DataFrame(arr, columns=['Keypoint1_x', 'Keypoint1_y', 'Keypoint2_x', 'Keypoint2_y',
                                              'Keypoint3_x', 'Keypoint3_y', 'Keypoint4_x', 'Keypoint4_y',
                                              'Keypoint5_x', 'Keypoint5_y', 'Keypoint6_x', 'Keypoint6_y',
                                              'Keypoint7_x', 'Keypoint7_y', 'Keypoint8_x', 'Keypoint8_y',
                                              'Keypoint9_x', 'Keypoint9_y', 'Image'])
        df = df.append(array_df)

    df = df.dropna(how='all')
    print(df)
    df = shuffle(df)
    trainDf, testDf = train_test_split(df, test_size=0.1)
    trainDf.to_pickle("./train_data.pkl")
    testDf.to_pickle("./test_data.pkl")


load()
