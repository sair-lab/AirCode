from __future__ import print_function
import sys
sys.path.append('.')
import cv2
import numpy as np
import math
import random

""" Data augmentation for gcn masks """

augmentations = [
    'additive_gaussian_noise',
    'additive_speckle_noise',
    'random_brightness',
    'random_contrast',
    'affine_transform',
    'perspective_transform',
    'random_crop',
    'add_shade',
    'motion_blur'
]

def erode(image, kernel_size):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
  image = cv2.erode(image, kernel)   
  return image

def dilate(image, kernel_size):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
  image = cv2.dilate(image, kernel)  
  return image

def random_region_zero(image, scale_x=0.3, scale_y=0.3):
  ys, xs = np.where(image > 0)
  x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
  
  region_width = (x1 - x0) * scale_x
  region_height = (y1 - y0) * scale_y

  x0 = random.uniform(x0, (x1 - region_width))
  y0 = random.uniform(y0, (y1 - region_height))

  x1 = x0 + region_width
  y1 = y0 + region_height

  x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

  image[y0:y1, x0:x1] = 0

  return image

def random_block_zero(image, num=5, size=10):
  ys, xs = np.where(image > 0)
  x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
  
  block_xs = random.sample(range(x0, x1), num)
  block_ys = random.sample(range(y0, y1), num)
  mask = np.ones_like(image)

  for y, x in zip(block_ys, block_xs):
    mask[y, x] = 0

  kernel_size = size
  mask = erode(mask, kernel_size)

  image = (image * mask).astype(np.uint8)

  return image

def random_block_one(image, num=10, size=10):
  H, W = image.shape[-2:]

  block_xs = random.sample(range(0, (W-1)), num)
  block_ys = random.sample(range(0, (H-1)), num)
  
  mask = np.zeros_like(image)

  for y, x in zip(block_ys, block_xs):
    mask[y, x] = 1

  kernel_size = size 
  mask = dilate(mask, kernel_size)
  
  img = (image > 0).astype(float)
  value = np.sum(image) / np.sum(img)

  img = img + mask

  img = (img > 0).astype(float) 

  img = img * value
  img = img.astype(np.uint8)

  return img


if __name__ == "__main__":

  from debug_tools.show_batch import show_numpy

  img1 = np.ones([640, 640])
  img2 = np.zeros([640, 640])
  img = np.concatenate([img1, img2], 1)
  img = (img * 150.0 + 0.5).astype(np.uint8)
  img = np.clip(img, 0, 255)

  img = dilate(img, kernel_size=10)

  img = cv2.merge([img, img, img])
  show_numpy(img)