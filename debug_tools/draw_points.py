#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages/')
import numpy as np
import time
import sys
import os
import argparse
from datetime import datetime
import yaml
import cv2

from datasets.utils.pipeline import makedir

data_root = "/home/haoyuefan/xk_data/superpoint/coco/full/coco"
images_dir = os.path.join(data_root, "train2014")
points_dir = os.path.join(data_root, "train2014_points")
debug_dir = os.path.join(data_root, "show/points")
makedir(debug_dir)

image_names = os.listdir(images_dir)
for image_name in image_names:
  image_path = os.path.join(images_dir, image_name)
  vis_image = os.path.join(debug_dir, image_name)
  file_name = image_name.split('.')[0]
  point_path = os.path.join(points_dir, '{}.txt'.format(file_name))

  img = cv2.imread(image_path)
  points = np.loadtxt(point_path, ndmin=2)
  for j in range(points.shape[0]):
    x = points[j][0].astype(int)
    y = points[j][1].astype(int)
    if(x < 0) :
      break
    cv2.circle(img, (y,x), 1, (0,0,255), thickness=-1)
  
  cv2.imwrite(vis_image, img)
