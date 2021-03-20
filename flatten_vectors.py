import numpy as np
from os import listdir
_image_size = 96

"""
This script assumes that all photos are reshaped to size 96 x 96
"""

data_path = r'./Ironman_cropped/Cropped/Cropped/Cropped_min/'
files = [f for f in listdir(data_path)]

for file in files:
    keypoints_path = data_path + file + '/'+file + '_keypoints'
    keypoints = np.loadtxt(keypoints_path)
    num_images = keypoints.shape[0]
    flatten = np.zeros((num_images, 9))
    for i in range(0, num_images):
        row = keypoints[i]
        for idx in range(0, 9):
                flatten[i, idx] = row[(idx*2+1)]*_image_size + row[idx*2]

    np.savetxt(keypoints_path+'flatten.txt', flatten,fmt='%i')