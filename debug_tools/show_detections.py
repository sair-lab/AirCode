#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')   
import datetime
import logging
import os
import time
import argparse
import yaml
import cv2
import torch
import torch.distributed as dist
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from model.build_model import build_maskrcnn, build_superpoint_model
from model.inference import detection_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_image(img_path):
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if len(image.shape) == 2:
    image = cv2.merge([image, image, image])
  return image

def show_detections(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  ## superpoint model config
  superpoint_model_config = configs['model']['superpoint']
  detection_threshold = superpoint_model_config['detection_threshold']
  gaussian_radius = 2
  ## others
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # model 
  maskrcnn_model = build_maskrcnn(configs)
  superpoint_model = build_superpoint_model(configs)

  # data
  image_names = os.listdir(data_root)

  transform = transforms.ToTensor()
  with torch.no_grad():
    maskrcnn_model.eval()    
    for image_name in image_names:
      print(image_name)
      image_path = os.path.join(data_root, image_name)
      image = read_image(image_path)
      image = transform(image)
      image = image.unsqueeze(0)
      batch = {'image': image, 'image_name': [image_name]}

      detection_inference(maskrcnn_model, superpoint_model, batch, use_gpu, 1,
          detection_threshold, data_config, save_dir=save_dir)


def main():
  parser = argparse.ArgumentParser(description="Training")
  parser.add_argument(
      "-c", "--config_file",
      dest = "config_file",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-g", "--gpu",
      dest = "gpu",
      type = int, 
      default = 0 
  )
  parser.add_argument(
      "-s", "--save_dir",
      dest = "save_dir",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-d", "--data_root",
      dest = "data_root",
      type = str, 
      default = "" 
  )
  parser.add_argument(
      "-mm", "--maskrcnn_model_path",
      dest = "maskrcnn_model_path",
      type = str, 
      default = "" 
  )
  parser.add_argument(
      "-sm", "--superpoint_model_path",
      dest = "superpoint_model_path",
      type = str, 
      default = "" 
  )
  args = parser.parse_args()

  config_file = args.config_file
  f = open(config_file, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)
  configs['use_gpu'] = args.gpu
  configs['save_dir'] = args.save_dir
  configs['data_root'] = args.data_root
  configs['maskrcnn_model_path'] = args.maskrcnn_model_path
  configs['superpoint_model_path'] = args.superpoint_model_path

  show_detections(configs)

if __name__ == "__main__":
    main()


    