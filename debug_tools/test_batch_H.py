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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import cv2

from models.superpoint import SuperPoint
from models.backbone.fcn import VGGNet
from models.vgg_like import VggLike
from models.superobject import SuperObject
from datasets.utils.pipeline import makedir
from debug_tools.show_batch import show_batch, show_numpy
from datasets.testdata.dataloader import TestDataset
from utils.tools import process_point, tensor_to_numpy
from datasets.utils.homographies import sample_homography, warp_batch_images

def export_points(configs):
  # read configs
  val_batch_size = configs['model']['batch_size']
  data_root = configs['data_root']
  cell = configs['model']['cell']
  img_new_size = configs['img_new_size']

  # dataset
  val_data = TestDataset(dataroot=data_root, img_new_size=img_new_size)
  val_loader = DataLoader(val_data, batch_size=val_batch_size, num_workers=8)

  for iter, batch in enumerate(val_loader):
    inputs = batch['image']
    img_shape = inputs.shape[-2:]
    H = sample_homography(img_shape, **configs['model']['homography_adaptation']['homographies'])
    show_batch(inputs)
    warped_img = warp_batch_images(inputs, H)
    show_batch(warped_img)

    img = tensor_to_numpy(inputs[0])
    img = cv2.warpPerspective(img, H, (img_shape[1], img_shape[0]))
    show_numpy(img)

def main():
  parser = argparse.ArgumentParser(description="export points")
  parser.add_argument(
      "-c", "--config_file",
      dest = "config_file",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-d", "--data_root",
      dest = "data_root",
      type = str, 
      default = "" 
  )
  args = parser.parse_args()

  config_file = args.config_file
  f = open(config_file, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)
  configs['data_root'] = args.data_root
  configs['img_new_size'] = [240, 320]

  export_points(configs)

if __name__ == "__main__":
    main()
