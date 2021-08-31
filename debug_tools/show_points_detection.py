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

from model.build_model import build_maskrcnn
from model.inference import maskrcnn_inference
from datasets.utils.pipeline import makedir
from debug_tools.show_batch import show_batch, show_numpy
from utils.tools import tensor_to_numpy
from datasets.utils.postprocess import nms_fast
from kornia.feature import match_nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def process(prob, border_remove, nms_dist):
  # Convert pytorch -> numpy.
  heatmap = prob.squeeze() # H * W
  ys, xs = np.where(heatmap > 0) # Confidence threshold.
  if len(xs) == 0:
    return None, None
  pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
  pts[0, :] = xs
  pts[1, :] = ys
  pts[2, :] = heatmap[ys, xs]
  H, W = heatmap.shape[-2:]
  pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)
  inds = np.argsort(pts[2,:])
  pts = pts[:,inds[::-1]] # Sort by confidence.
  # Remove points along border.
  bord = border_remove
  toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
  toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
  toremove = np.logical_or(toremoveW, toremoveH)
  pts = pts[:, ~toremove]
  
  return pts

def extract_points(img_path, model, detection_thr):

  transform = transforms.ToTensor()

  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if len(image.shape) == 2:
    image = cv2.merge([image, image, image])

  sizes = {}
  original_image_sizes = [image.shape[-2:]]
  sizes['original_sizes'] = torch.tensor(original_image_sizes)
  sizes['new_sizes'] = torch.tensor(original_image_sizes)

  img = transform(image)
  img = img.unsqueeze(0)
  _, _, points_output = model(img, sizes)
  
  # process point
  prob = points_output['prob'].cpu().detach().gt(detection_thr).float().numpy()
  points = process(prob, 4, 4)

  return image, points

def draw_points(points, img, color=(255,0,0)):
  for j in range(points.shape[1]):
    x = points[0][j].astype(int)
    y = points[1][j].astype(int)
    if x < 0:
      break
    cv2.circle(img, (x,y), 1, color, thickness=-1)
  return img

def show_image_points(dataroot, image_name, save_dir, model, detection_thr):
  with torch.no_grad():
    model.eval()    

    img_path = os.path.join(dataroot, image_name)
    img, points = extract_points(img_path, model, detection_thr)
    h, w = img.shape[:2]
    img = draw_points(points, img)

    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, img)

def show_points(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  superpoint_model_config = configs['model']['superpoint']
  detection_thr = superpoint_model_config['eval']['detection_threshold']
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # model 
  maskrcnn_model = build_maskrcnn(configs)

  image_names = os.listdir(data_root)

  for image_name in image_names:
    show_image_points(data_root, image_name, save_dir, maskrcnn_model, detection_thr)


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
      "-m", "--model_path",
      dest = "pretrained_model_path",
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
  configs['pretrained_model_path'] = args.pretrained_model_path
  configs['save_dir'] = args.save_dir

  show_points(configs)

if __name__ == "__main__":
  main()
