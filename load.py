from sklearn.utils import shuffle
import os
import pandas as pd
from glob import glob
import cv2
import numpy as np

DATA_PATH = r'./Dataset'


def load(test=False, width=96, height=96, sigma=5):
    pass

    # df = pd.read_csv(os.path.expanduser(fname))
    #
    # df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    #
    # myprint = df.count()
    # myprint = myprint.reset_index()
    # print(myprint)
    # ## row with at least one NA columns are removed!
    # ## df = df.dropna()
    # df = df.fillna(-1)
    #
    # X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    # X = X.astype(np.float32)
    #
    # if not test:  # labels only exists for the training data
    #     y, y0, nm_landmark = get_y_as_heatmap(df, height, width, sigma)
    #     X, y, y0 = shuffle(X, y, y0, random_state=42)  # shuffle data
    #     y = y.astype(np.float32)
    # else:
    #     y, y0, nm_landmark = None, None, None
    #
    # return X, y, y0, nm_landmark


def load2d(test=False, width=96, height=96, sigma=5):
    df = pd.DataFrame(columns=['Keypoint1_x', 'Keypoint1_y', 'Keypoint2_x', 'Keypoint2_y',
                              'Keypoint3_x', 'Keypoint3_y', 'Keypoint4_x', 'Keypoint4_y',
                              'Keypoint5_x', 'Keypoint5_y', 'Keypoint6_x', 'Keypoint6_y',
                              'Keypoint7_x', 'Keypoint7_y', 'Keypoint8_x', 'Keypoint8_y',
                              'Keypoint9_x', 'Keypoint9_y', 'Image'])

    subdirs = [ DATA_PATH + '/'+ name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name)) ]

    for subdir in subdirs:
        keypoint_file = glob(subdir + '/*keypoints')
        file = open(keypoint_file[0], "r")
        file_content = file.read()
        arr = [line.split() for line in file_content.split('\n')]

        color_subdir = os.path.join(subdir, 'color')
        for i, filename in enumerate(os.listdir(color_subdir)):
            image = np.array(cv2.imread(os.path.join(color_subdir, filename), cv2.IMREAD_GRAYSCALE))
            image = image / 255
            arr[i].append(image)

        array_df = pd.DataFrame(arr, columns=['Keypoint1_x', 'Keypoint1_y', 'Keypoint2_x', 'Keypoint2_y',
                              'Keypoint3_x', 'Keypoint3_y', 'Keypoint4_x', 'Keypoint4_y',
                              'Keypoint5_x', 'Keypoint5_y', 'Keypoint6_x', 'Keypoint6_y',
                              'Keypoint7_x', 'Keypoint7_y', 'Keypoint8_x', 'Keypoint8_y',
                              'Keypoint9_x', 'Keypoint9_y', 'Image'])
        df = df.append(array_df)

    print(df)

    # re = load(test, width, height, sigma)
    # X = re[0].reshape(-1, width, height, 1)
    # y, y0, nm_landmarks = re[1:]
    #
    # return X, y, y0, nm_landmarks

load2d()