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
import yaml

from datasets.utils.pipeline import makedir

class KittiOdometry(Dataset):
  def __init__(self, data_root, id, dis_thr=15, angle_thr=1.0, interval=100):
    image_dir = os.path.join(data_root, "images", id, "image_0")
    label_file = os.path.join(data_root, "poses", (id+".txt"))
    image_names = os.listdir(image_dir)
    image_names.sort()


    self.data_root = data_root
    self.id = id
    self.dis_thr = dis_thr
    self.angle_thr = angle_thr
    self.interval = interval
    self.image_dir = image_dir
    self.image_names = image_names
    self.length = len(image_names)
    self.poses_gt = self.read_label_file(label_file)
    loop_gt, num_loop = self.find_loops(dis_thr, angle_thr, interval)
    # loop_gt, num_loop = None, None
    self.loop_gt = loop_gt
    self.num_loop = num_loop
    self.transform = transforms.ToTensor()

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_names[idx])  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if len(image.shape) == 2:
      image = cv2.merge([image, image, image])
      
    return {'image': image, 'image_name': self.image_names[idx]}


  def read_label_file(self, file_path):
    poses_gt = {}

    index = 0
    fo = open(file_path, "r")
    for line in fo.readlines():
      line = line.strip('\n')
      line = line.split(' ')  
      line = [float(l) for l in line]
      line = np.array(line)
      line = line.reshape(3, 4)
      position = line[:, -1]
      rotation = line[:, :3]

      gt = {'index':index, 'position':position, 'rotation':rotation}

      image_name = self.image_names[index]
      poses_gt[image_name] = gt
      index = index + 1

    fo.close()

    return poses_gt


  def get_label(self, r):
    if type(r) == type(0):
      image_name = self.image_names[r]
    else:
      image_name = r
    
    if image_name in self.poses_gt:
      pose_gt = self.poses_gt[image_name]
    else:
      pose_gt = None
    
    return pose_gt


  def find_loops(self, dis_thr, angle_thr, interval):
    loop_gt = {}
    num_loop = 0
    for i in range(len(self.image_names)):
      image_name = self.image_names[i]
      if i < interval:
        loop_gt[image_name] = 0
        continue

      gt_i = self.get_label(i)
      position_i = gt_i['position']
      rotation_i = gt_i['rotation']

      for j in range(i):
        if i - j < interval:
          loop_gt[image_name] = 0
          break
      
        gt_j = self.get_label(j)
        position_j = gt_j['position']
        rotation_j = gt_j['rotation']

        delta_dis = np.linalg.norm((position_i-position_j))
        delta_R = rotation_i.dot(rotation_j.T)
        delta_r, _ = cv2.Rodrigues(delta_R)
        deleta_angle = np.linalg.norm(delta_r)

        if delta_dis < dis_thr and deleta_angle < angle_thr:
          loop_gt[image_name] = 1
          num_loop += 1
          break

    return loop_gt, num_loop


  def get_loop_gt(self):
    return self.loop_gt, self.num_loop

  def image_size(self):
    '''
    H, W
    '''
    image_path = os.path.join(self.image_dir, self.image_names[0])  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.shape[-2:]
