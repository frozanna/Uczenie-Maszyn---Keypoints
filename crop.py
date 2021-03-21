import numpy as np
import os
import glob                     
import cv2

folder = "./Dataset/Train/"
category = "ir_or_5_f/"
color = "color/"
mask = "mask/"

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

color_paths = load_images_from_folder(folder, category+color)
mask_paths = load_images_from_folder(folder, category+mask)

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
        
print('done')        
print('\n')