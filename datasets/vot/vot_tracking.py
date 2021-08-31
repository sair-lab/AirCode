#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import numbers
import random
import os
import cv2

class VotTracking(Dataset):
  def __init__(self, data_root, id):
    image_dir = os.path.join(data_root, id, "images")
    label_file = os.path.join(data_root, id, "configs/groundtruth.txt")
    image_names = os.listdir(image_dir)
    image_names.sort()

    self.image_dir = image_dir
    self.image_names = image_names
    self.length = len(image_names)
    self.track_gt = self.read_label_file(label_file)
    self.transform = transforms.ToTensor()

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_names[idx])  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if len(image.shape) == 2:
      image = cv2.merge([image, image, image])
      
    # image = self.transform(image)

    return {'image': image, 'image_name': self.image_names[idx]}

  def read_label_file(self, file_path):
    track_gt = {}

    fo = open(file_path, "r")
    i = 0
    for line in fo.readlines():
      line = line.strip('\n')
      line = line.split(',')     
      track_id = 0
      frame_id = i
      # x1, y1, w, h
      if len(line) == 4:
        x1, y1, w, h = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        x2, y2 = (x1 + w - 1), (y1 + h - 1) 
      else:
        assert len(line) == 8
        line = [float(l) for l in line]
        line = np.array(line)
        x1, x2 = min(line[0::2]), max(line[0::2])
        y1, y2 = min(line[1::2]), max(line[1::2])

      box = [x1, y1, x2, y2]
      object_info = {'frame_id':frame_id, 'track_id':track_id, 'box':box, }
      i = i + 1
      image_name = self.image_names[frame_id]
      track_gt[image_name] = object_info

    fo.close()

    return track_gt

  def get_label(self, r):
    if type(r) == type(0):
      image_name = self.image_names[r]
    else:
      image_name = r
    
    if image_name in self.track_gt:
      image_info = self.track_gt[image_name]
    else:
      image_info = None
    
    return image_info

  def image_size(self):
    '''
    H, W
    '''
    image_path = os.path.join(self.image_dir, self.image_names[0])  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.shape[-2:]
