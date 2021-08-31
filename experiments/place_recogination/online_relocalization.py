#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os

import torch
from torchvision import transforms
import yaml
import cv2
import numpy as np
import argparse

from model.build_model import build_maskrcnn, build_gcn, build_superpoint_model
from datasets.utils.pipeline import makedir
from datasets.kitti.kitti_odomery import KittiOdometry

from experiments.object_tracking.object_tracking import update_normal_size, network_output, calculate_pr_curve, box_iou

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def relocalization(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  model_dir = configs['model_dir']
  dataset_name = configs['data']['name']
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  superpoint_model_path = os.path.join(model_dir, "points_model.pth")
  maskrcnn_model_path = os.path.join(model_dir, "maskrcnn_model.pth")
  gcn_model_path = os.path.join(model_dir, "gcn_model.pth")
  configs["maskrcnn_model_path"] = maskrcnn_model_path
  configs["superpoint_model_path"] = superpoint_model_path
  configs["graph_model_path"] = gcn_model_path

  # model 
  superpoint_model = build_superpoint_model(configs, requires_grad=False)
  superpoint_model.eval()

  maskrcnn_model = build_maskrcnn(configs)
  maskrcnn_model.eval()

  gcn_model = build_gcn(configs)
  gcn_model.eval()

  # data
  seqs = ['00', '05', '06']
  
  pr_curves_list = []
  for seq in seqs:
    dataset = KittiOdometry(data_root, seq)
    dis_thr = dataset.dis_thr
    angle_thr = dataset.angle_thr
    interval = dataset.interval
    
    image_size = dataset.image_size()
    configs['data']['normal_size'] = update_normal_size(image_size)
    
    seq_save_dir = os.path.join(save_dir, seq)
    makedir(seq_save_dir)
    for data in dataset:
      image = data['image']
      image_name = data['image_name']
      print(image_name)

      net_output = network_output(image, superpoint_model, maskrcnn_model, gcn_model, configs)
      net_output = {'points': net_output[0], 'objects': net_output[1], 'descs': net_output[2]}
      if net_output['points'] is None:
        continue

      image_save_dir = os.path.join(seq_save_dir, image_name)
      makedir(image_save_dir)
      
      # save points
      points_dir = os.path.join(image_save_dir, "points")
      makedir(points_dir)
      points = net_output['points']
      for i in range(len(points)):
        p = points[i].cpu().numpy
        p_path = os.path.join(points_dir, (str(i) + ".npy"))
        np.save(p_path, p)

      # save descs
      descs_path = os.path.join(image_save_dir, "descs.npy")
      descs = net_output['descs'].cpu().numpy()
      np.save(descs_path, descs)

      # save objects
      objects = net_output['objects']
      objects_dir = os.path.join(image_save_dir, "objects")
      makedir(objects_dir)
      for k in objects.keys():
        value = objects[k].cpu().numpy()
        value_path = os.path.join(objects_dir, (k+".npy"))
        np.save(value_path, value)



def main():
  parser = argparse.ArgumentParser(description="show match")
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
      "-m", "--model_dir",
      dest = "model_dir",
      type = str, 
      default = "" 
  )
  args = parser.parse_args()
  config_file = args.config_file
  f = open(config_file, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)
  configs['use_gpu'] = args.gpu
  configs['data_root'] = args.data_root
  configs['model_dir'] = args.model_dir
  configs['save_dir'] = args.save_dir

  relocalization(configs)

if __name__ == "__main__":
  main()
