import numpy as np
import os
import glob                
import cv2

folder = "./Dataset/Test/"
category = "ir_fm_f/"
keypoint = "ir_fm_f_keypoints"
color = "color/"
mask = "mask/"
rsize = 96

def crop_mask(img):
    mask = img
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    col_diff = col_end-col_start
    row_diff = row_end-row_start
    if(col_diff > row_diff):
        diff = col_diff - row_diff
        if( (diff % 2) != 0):
            row_end = row_end + 1
            diff = diff - 1
        row_start = row_start - int(diff/2)
        row_end = row_end + int(diff/2)
    elif(col_diff < row_diff):
        diff = row_diff - col_diff
        if( diff % 2 != 0):
            col_end = col_end + 1
            diff = diff - 1
        col_start = col_start - int(diff/2)
        col_end = col_end + int(diff/2)
    return img[row_start:row_end,col_start:col_end], col_start, col_end, row_start, row_end

def crop_image(img, col_start, col_end, row_start, row_end):
    return img[row_start:row_end,col_start:col_end]

def load_images_from_folder(f, c):
    images = []
    for filename in os.listdir(folder + c):
        img = f + c + filename
        if img is not None:
            images.append(filename)
    return images

def resize_img_and_keypint(img, n):
    img_resized = img.resize((96, 96))
    ratio = 96 / img.height


color_paths = load_images_from_folder(folder, category+color)
mask_paths = load_images_from_folder(folder, category+mask)
keypoint_file = folder + category + keypoint
# print(keypoint_file)
file = open(keypoint_file, "r")
file_content = file.read()
keypoints = [line.split() for line in file_content.split('\n')]
# print(keypoints[0][:])
# ratio = 96 / 192
# mask_org = cv2.imread(folder + category + mask + mask_paths[i], 0)
# print()
new_keypoints = []
# for j in range(len(keypoints)):
#     col = []
#     for k in range(len(keypoints[j])):
#         col.append(int(round(int(keypoints[0][k])*ratio)))
#         # print(int(round(int(keypoints[0][k])*ratio)))
#     new_keypoints.append(col)
# print(new_keypoints)
for i in range(len(mask_paths)):
    mask_org = cv2.imread(folder + category + mask + mask_paths[i], 0)
    img_org = cv2.imread(folder + category + color + color_paths[i])

    mask_crop, col_start, col_end, row_start, row_end = crop_mask(mask_org)
    img_crop = crop_image(img_org, col_start, col_end, row_start, row_end)

    writePathColor = folder + 'Cropped/' + category + color
    writePathMask = folder + 'Cropped/' + category + mask
    if not os.path.exists(writePathMask):
        os.makedirs(writePathMask)
    if not os.path.exists(writePathColor):
        os.makedirs(writePathColor)
    
    cv2.imwrite(os.path.join(writePathMask, mask_paths[i]), mask_crop)
    cv2.imwrite(os.path.join(writePathColor, color_paths[i]), img_crop)
    
    mask_resized = cv2.resize(mask_crop, dsize=(rsize, rsize), interpolation=cv2.INTER_CUBIC)
    img_resized = cv2.resize(img_crop, dsize=(rsize, rsize), interpolation=cv2.INTER_CUBIC)
    writePathColorRS = folder + 'Cropped_96/' + category + color
    writePathMaskRS = folder + 'Cropped_96/' + category + mask
    if not os.path.exists(writePathMaskRS):
        os.makedirs(writePathMaskRS)
    if not os.path.exists(writePathColorRS):
        os.makedirs(writePathColorRS)
    
    cv2.imwrite(os.path.join(writePathMaskRS, mask_paths[i]), mask_resized)
    cv2.imwrite(os.path.join(writePathColorRS, color_paths[i]), img_resized)
    
    w = len(img_crop)
    ratio = rsize / w
    col = []
    for k in range(len(keypoints[i])):
        col.append(int(round(int(keypoints[i][k])*ratio)))
    new_keypoints.append(col)

new_keypoints_file = folder + 'Cropped_96/' + category + keypoint + "_" + str(rsize)
f = open(new_keypoints_file, "w")
np.savetxt(f, new_keypoints, fmt='%s', delimiter='\t', newline='\n')
        
print('done')        
print('\n')