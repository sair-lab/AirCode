# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import argparse
import yaml

import torch
import torch.distributed as dist

import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from model.mask_rcnn.mask_rcnn import MaskRCNN
from datasets.utils.build_data import coco_loader
from model.build_model import build_maskrcnn, build_superpoint_model
from model.inference import detection_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def validate(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  val_data_name = data_config['VAL']
  ## superpoint model config
  superpoint_model_config = configs['model']['superpoint']
  detection_threshold = superpoint_model_config['detection_threshold']
  val_batch_size = superpoint_model_config['batch_size']
  gaussian_radius = 2
  ## others
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # data
  val_loader = coco_loader(data_root=data_root, name=val_data_name, config=data_config, 
      batch_size=val_batch_size, remove_images_without_annotations=True)

  # model 
  maskrcnn_model = build_maskrcnn(configs)
  superpoint_model = build_superpoint_model(configs)

  with torch.no_grad():
    maskrcnn_model.eval()    
    for iter, batch in enumerate(val_loader):
      result = detection_inference(maskrcnn_model, superpoint_model, batch, use_gpu, gaussian_radius,
          detection_threshold, data_config, save_dir)

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

  validate(configs)

if __name__ == "__main__":
    main()


    