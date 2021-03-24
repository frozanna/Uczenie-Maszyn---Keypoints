import numpy as np
import os
import PIL
from PIL import Image
from  scipy import ndimage
import scipy as sc
import math
from numpy import asarray
import glob



def coord(lines):
  list_of_keypoints=[]
  for j in range(len(lines)):
    s=lines[j].strip()
    k = s.split(" ")
    k = list(filter(None, k))
    kp=[]
    for i in range(0,len(k),2):
      kp.append((int(k[i]), int(k[i+1])))
    list_of_keypoints.append(kp)
  return list_of_keypoints
  
def mark_img(img, kp):
  img1=img.copy()
  for p in kp:
    # for j in range(img1.shape[2]):
    img1[p[1]][p[0]]=[255, 0, 0]
  return img1



def rot(origin, point, angle):
    angle = math.radians(-angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (round(qx), round(qy))

def key_points_map(shape, kp, color=(255,0,0)):
  kp_map = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8)
  kp_map.fill(255)
  for i in range(len(kp)):
    kp_map[kp[i][1]][kp[i][0]]=[color[0], color[1], color[2]]
  return kp_map




def fliprl(img, kp):
  shape = img.shape
  img_new = np.fliplr(img)
  kp_new = list(map(lambda x: (shape[0]-x[0], x[1]), kp))
  return img_new, kp_new

def flipud(img, kp):
  shape = img.shape
  img_new = np.flipud(img)
  kp_new = list(map(lambda x: (x[0], shape[1] - x[1]), kp))
  return img_new, kp_new 

def fliprlud(img, kp):
  shape = img.shape
  img_new = np.fliplr(img)
  img_new = np.flipud(img_new)
  kp_new = list(map(lambda x: (shape[0]-x[0], shape[1] - x[1]), kp))
  return img_new, kp_new

def rotate(img, kp, angle=45, mode="reflect"):
  kp_pom=[]
  new_image=ndimage.rotate(img, angle, reshape=False, mode=mode)
  kp_new = list(map(lambda x: rot((img.shape[1]/2, img.shape[0]/2), x, angle), kp))
  kp_new = list(filter(lambda x: x[0]>=0 and x[0]<=img.shape[1] and x[1]>=0 and x[1]<=img.shape[0], kp_new))
  if len(kp_new)==9:
    return new_image, kp_new
  else:
    return None, None


filename = "Dataset/ir_fm_f/ir_fm_f_keypoints"
f = open(filename, 'r')
lines = f.readlines()
f.close()
rot_angles=[45, 135, 225, 315]
#rot_angles=[]

g=glob.glob("Dataset/ir_fm_f/color/*.png")
#g=list(map( lambda x: x[22:], g))
d = list(zip(list(map(lambda x: int(x[22+4:-4]), g)), g ))
d.sort()
d = list(zip(d, coord(lines)))
d=list(map(lambda x: (x[0][0], x[0][1], x[1]), d))
last = d[-1][0]

for j in range(len(d)):
  image = Image.open(d[j][1])
  img = asarray(image)
  kp=d[j][2]

  i, k= fliprl(img, kp)
  if not (i is None or k is None):
    img2 = Image.fromarray(i)
    last+=1
    img2.save("Dataset/ir_fm_f/color/img_"+str(last).zfill(6)+".png")
    f = open(filename, "a")
    #k=list(map(lambda x: (x[1], x[0]), k))
    f.write(' %d  %d   %d  %d  %d  %d  %d  %d  %d   %d  %d  %d  %d  %d  %d   %d  %d  %d\n' %tuple([e for l in k for e in l]))
    f.close()

  i, k= flipud(img, kp)
  if not (i is None or k is None):
    img2 = Image.fromarray(i)
    last+=1
    img2.save("Dataset/ir_fm_f/color/img_"+str(last).zfill(6)+".png")
    f = open(filename, "a")
   # k=list(map(lambda x: (x[1], x[0]), k))
    f.write(' %d  %d   %d  %d  %d  %d  %d  %d  %d   %d  %d  %d  %d  %d  %d   %d  %d  %d\n' % tuple([e for l in k for e in l]))
    f.close()

  i, k= fliprlud(img, kp)
  if not (i is None or k is None):
    img2 = Image.fromarray(i)
    last+=1
    img2.save("Dataset/ir_fm_f/color/img_"+str(last).zfill(6)+".png")
    f = open(filename, "a")
  #  k=list(map(lambda x: (x[1], x[0]), k))
    f.write(' %d  %d   %d  %d  %d  %d  %d  %d  %d   %d  %d  %d  %d  %d  %d   %d  %d  %d\n' % tuple([e for l in k for e in l]))
    f.close()
    
  if len(rot_angles)>0:
    for angle in rot_angles:
      i, k= rotate(img, kp, angle=angle)
      if not (i is None or k is None):
        img2 = Image.fromarray(i)
        last+=1
        img2.save("Dataset/ir_fm_f/color/img_"+str(last).zfill(6)+".png")
        f = open(filename, "a")
      #  k=list(map(lambda x: (x[1], x[0]), k))
        f.write(' %d  %d   %d  %d  %d  %d  %d  %d  %d   %d  %d  %d  %d  %d  %d   %d  %d  %d\n' % tuple([e for l in k for e in l]))
        f.close()
        

