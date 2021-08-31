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

from model.build_model import build_superpoint_model
from model.inference import maskrcnn_inference
from datasets.utils.pipeline import makedir
from debug_tools.show_batch import show_batch, show_numpy
from utils.tools import tensor_to_numpy
from datasets.utils.postprocess import nms_fast
from kornia.feature import match_nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def process(prob, desc, border_remove, nms_dist):
  # Convert pytorch -> numpy.
  heatmap = prob.squeeze() # H * W
  desc_data = desc.squeeze() # 256 * H * W
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

  desc_point = []
  for i in range(pts.shape[1]):
    xs = int(pts[0][i])
    ys = int(pts[1][i])
    desc_point = desc_point + [desc_data[:, ys, xs]]
  
  desc_point = np.stack(desc_point)
  return pts, desc_point

def extract_desc(img_path, model, detection_thr, img_new_size):

  transform = transforms.ToTensor()

  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  image = cv2.resize(image, tuple(img_new_size[::-1]), interpolation=cv2.INTER_LINEAR)
  if len(image.shape) == 2:
    image = cv2.merge([image, image, image])

  img = transform(image)
  img = img.unsqueeze(0)
  points_output = model(img)
  
  # process point
  prob = points_output['prob'].cpu().detach().gt(detection_thr).float().numpy()
  desc = points_output['desc'].cpu().detach().float().numpy()
  points, descs = process(prob, desc, 4, 4)

  return image, points, descs

def draw_points(points, img, color=(255,0,0)):
  for j in range(points.shape[1]):
    x = points[0][j].astype(int)
    y = points[1][j].astype(int)
    if x < 0:
      break
    cv2.circle(img, (x,y), 1, color, thickness=-1)
  return img

def generate_pair_result(dataroot, name, save_dir, model, detection_thr, img_new_size):
  with torch.no_grad():
    model.eval()    

    pair_path = os.path.join(dataroot, name)
    image_names = os.listdir(pair_path)

    img1_path = os.path.join(pair_path, image_names[0])
    img2_path = os.path.join(pair_path, image_names[1])

    img1, points1, desc1 = extract_desc(img1_path, model, detection_thr, img_new_size)
    img2, points2, desc2 = extract_desc(img2_path, model, detection_thr, img_new_size)

    h, w = img1.shape[:2]

    desc1 = torch.tensor(desc1)
    desc2 = torch.tensor(desc2)

    dis, match = match_nn(desc1, desc2)
    dis, match = dis.squeeze(1).numpy(), match.numpy()
    img1 = draw_points(points1, img1)
    img2 = draw_points(points2, img2, (0, 255, 0))

    img = np.concatenate([img1, img2], 1)
    for i in range(match.shape[0]):
      if dis[i] > 0.7 :
        continue

      idx1 = int(match[i, 0])
      idx2 = int(match[i, 1])

      px1, py1 = int(points1[0][idx1]), int(points1[1][idx1]) 
      px2, py2 = int(points2[0][idx2] + w), int(points2[1][idx2]) 

      p1 = (px1, py1)
      p2 = (px2, py2)

      a = np.random.randint(0,255)
      b = np.random.randint(0,255)
      c = np.random.randint(0,255)

      cv2.line(img, (px1, py1), (px2, py2), (a, b, c), 1)

    save_name = name + ".png"
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, img)

def show_match(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  superpoint_model_config = configs['model']['superpoint']
  detection_thr = superpoint_model_config['detection_threshold']
  img_new_size = configs['img_new_size']
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # model 
  superpoint_model = build_superpoint_model(configs, requires_grad=False)
  superpoint_model.eval()


  pair_names = os.listdir(data_root)

  for pair_name in pair_names:
    generate_pair_result(data_root, pair_name, save_dir, superpoint_model, detection_thr, img_new_size)


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
  configs['data_root'] = args.data_root
  configs['superpoint_model_path'] = args.superpoint_model_path
  configs['save_dir'] = args.save_dir
  configs['img_new_size'] = [480, 640]

  show_match(configs)

if __name__ == "__main__":
  main()
