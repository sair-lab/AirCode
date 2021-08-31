#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import numbers
import random
import os
import cv2

from datasets.utils.pipeline import draw_umich_gaussian
from datasets.utils import pipeline as pp

class SyntheticDataset(Dataset):
  def __init__(self, data_root, use_for = None):
    image_dir = os.path.join(data_root, use_for, "images")
    point_dir = os.path.join(data_root, use_for, "points")
    image_names = os.listdir(image_dir)

    self.image_dir = image_dir
    self.point_dir = point_dir
    self.image_names = image_names
    self.length = len(image_names)

    self.transform = transforms.ToTensor()

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_names[idx])  
    point_name = self.image_names[idx].split('.')[0] + ".txt"
    point_path = os.path.join(self.point_dir, point_name)  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    points = np.loadtxt(point_path, dtype=np.float32, ndmin=2)
    if np.sum(points) < 0:
      points = np.empty((0, 2), dtype=np.float32)
      
    image_name = self.image_names[idx].split('.')
    image_name = image_name[0]
    
    if len(image.shape) == 2:
      image = cv2.merge([image, image, image])
      
    image = self.transform(image)

    points = torch.tensor(points)

    return {'image': image, 'image_name': image_name, 'points': points}