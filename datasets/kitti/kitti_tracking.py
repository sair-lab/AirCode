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

class KittiTracking(Dataset):
  def __init__(self, data_root, id):
    image_dir = os.path.join(data_root, "images", id)
    label_file = os.path.join(data_root, "labels", (id+".txt"))
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
    for line in fo.readlines():
      line = line.strip('\n')
      line = line.split(' ')     
      track_id = int(line[1])
      if track_id < 0:
        continue 
      frame_id = int(line[0])
      object_type = line[2]
      truncated = int(line[3])
      occulded = int(line[4])
      # x1, y1, x2, y2
      box = [float(line[6]), float(line[7]), float(line[8]), float(line[9])]
      object_info = {'frame_id':frame_id, 'track_id':track_id, 'object_type':object_type, 
          'truncated':truncated, 'occulded':occulded, 'box':box, }

      image_name = self.image_names[frame_id]
      if image_name in track_gt:
        track_gt[image_name].append(object_info)
      else:
        track_gt[image_name] = [object_info]

    fo.close()

    # re-organioze groundtruth, Dict[List[Dict]] -> Dict[Dict[List]]
    new_track_gt = {}
    for image_name in track_gt.keys():
      if len(track_gt[image_name]) > 0:
        new_track_gt[image_name] = {}
        for k in track_gt[image_name][0].keys():
          data_list = [data[k] for data in track_gt[image_name]]
          new_track_gt[image_name][k] = data_list

    return new_track_gt

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
