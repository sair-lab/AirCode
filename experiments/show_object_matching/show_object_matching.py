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
import copy

from model.inference import maskrcnn_inference
from model.build_model import build_maskrcnn, build_gcn, build_superpoint_model
from model.inference import detection_inference
from datasets.utils.pipeline import makedir
from kornia.feature import match_nn
from datasets.utils import preprocess
from experiments.show_object_matching.draw_object import draw_object, compute_colors_for_labels

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def filter_objects(data, target_labels=None):
  '''
  data = {'image':image, 'points': output[0], 'objects': output[1],
      'descs': output[2], 'keeps': output[3]} 
  '''
  points = data['points']
  descs = data['descs']

  keeps = data['keeps']
  objects = data['objects']
  labels = objects['labels'][keeps]
  boxes = objects['boxes'][keeps]
  masks = objects['masks'][keeps]

  if target_labels is not None:
    new_labels, new_boxes, new_masks = [], [], []
    new_points, new_descs = [], []
    for i in range(len(points)):
      if labels[i].item() in target_labels:
        new_labels.append(labels[i])
        new_boxes.append(boxes[i])
        new_masks.append(masks[i])  
        new_points.append(points[i])  
        new_descs.append(descs[i])  
    
    labels = torch.stack(new_labels)
    boxes = torch.stack(new_boxes)
    masks = torch.stack(new_masks)
    points = new_points
    descs = torch.stack(new_descs)

  objects['labels'] = labels
  objects['boxes'] = boxes
  objects['masks'] = masks
  data['objects'] = objects

  data['points'] = points
  data['descs'] = descs
  return data


def draw_results(tpl_data, data, save_dir, image_name, match_thr=0.95):

  # get colors
  tpl_obj_num = len(tpl_data['objects']['boxes'])
  obj_num = len(data['objects']['boxes'])
  sum_num = tpl_obj_num + obj_num
  index = [(i+1)*30 for i in range(sum_num)]
  index = torch.tensor(index)
  sum_colors = compute_colors_for_labels(index).tolist()
  tpl_colors = sum_colors[:tpl_obj_num]
  colors = sum_colors[tpl_obj_num:sum_num]
  print(tpl_colors)

  # match
  tpl_descs = tpl_data['descs']
  descs = data['descs']
  dis, match = match_nn(tpl_descs, descs)
  dis, match = dis.cpu().squeeze(1).numpy(), match.cpu().numpy()
  print(dis)

  # update match colors
  match_idx_list1, match_idx_list2 = [], []
  for i in range(match.shape[0]):
    if dis[i] > match_thr :
      continue
    idx1 = int(match[i, 0])
    idx2 = int(match[i, 1])
    colors[idx2] = tpl_colors[idx1]
    match_idx_list1.append(idx1)
    match_idx_list2.append(idx2)

  # draw object
  tpl_image, _ = draw_object(tpl_data, tpl_colors, match_idx_list1)
  image, _ = draw_object(data, colors, match_idx_list2)
  img = np.concatenate([tpl_image, image], 1)
  # img = np.concatenate([tpl_image, image], 0)

  # draw match
  tpl_boxes = tpl_data['objects']['boxes']
  boxes = data['objects']['boxes']
  for i in range(match.shape[0]):
    if dis[i] > match_thr :
      continue

    idx1 = int(match[i, 0])
    idx2 = int(match[i, 1])

    c = tpl_colors[idx1]

    tpl_box = tpl_boxes[idx1]
    x1 = (int)((tpl_box[0] + tpl_box[2]) / 2)
    y1 = (int)((tpl_box[1] + tpl_box[3]) / 2)   
    cv2.circle(img, (x1, y1), 10, tuple(c), 2)

    box = boxes[idx2]
    x2 = (int)((box[0] + box[2]) / 2 + tpl_image.shape[-2])
    y2 = (int)((box[1] + box[3]) / 2)
    # x2 = (int)((box[0] + box[2]) / 2)
    # y2 = (int)((box[1] + box[3]) / 2  + tpl_image.shape[0])      
    cv2.circle(img, (x2, y2), 10, tuple(c), 2)

    cv2.line(img, (x1, y1), (x2, y2), tuple(c), 2)

  save_path = os.path.join(save_dir, image_name)
  cv2.imwrite(save_path, img)


def network_output(image, points_model, maskrcnn_model, gcn_model, configs, filter_labes=None):  
  with torch.no_grad():
    data_config = configs['data']
    superpoint_model_config = configs['model']['superpoint']
    detection_threshold = superpoint_model_config['detection_threshold']
    use_gpu = configs['use_gpu']

    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)
    batch = {'image': image}

    points_output, detections, _ = detection_inference(maskrcnn_model, points_model, batch, use_gpu, 1,
        detection_threshold, data_config, save_dir=None)

    batch_points, batch_descs = preprocess.extract_points_clusters(points_output, list([detections[0]['masks']]))

    original_sizes = [list(img.shape[-2:]) for img in image]

    batch_points = preprocess.normalize_points(batch_points, original_sizes)

    batch_points = preprocess.batch_merge(batch_points)
    batch_descs = preprocess.batch_merge(batch_descs)

    keeps = preprocess.select_good_clusters(batch_points)
    
    good_points, good_descs = [], []
    for i in range(len(keeps)):
      if keeps[i].item():
        good_points.append(batch_points[i])
        good_descs.append(batch_descs[i])
    
    batch_object_descs, _ = gcn_model(good_points, good_descs)

    return good_points, detections[0], batch_object_descs, keeps


def read_image(img_path):
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if len(image.shape) == 2:
    image = cv2.merge([image, image, image])
  return image

def show_object_matching(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  model_dir = configs['model_dir']
  use_gpu = configs['use_gpu']
  superpoint_model_config = configs['model']['superpoint']
  detection_thr = superpoint_model_config['detection_threshold']
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

  # template image
  tpl_path = os.path.join(data_root, "template.jpg")
  tpl_image = read_image(tpl_path)
  tpl_output = network_output(tpl_image, superpoint_model, maskrcnn_model, gcn_model, configs)
  tpl_data = {'image':tpl_image, 'points': tpl_output[0], 'objects': tpl_output[1],
       'descs': tpl_output[2], 'keeps': tpl_output[3]}


  # filter data
  target_labels = [40]
  # tpl_data = filter_objects(tpl_data, target_labels)
  tpl_labels = tpl_data['objects']['labels']
  print(tpl_labels)

  seq_path = os.path.join(data_root, "seq")
  image_names = os.listdir(seq_path)
  image_names.sort()
  with torch.no_grad():
    for image_name in image_names:
      image_path = tpl_path = os.path.join(seq_path, image_name)
      image = read_image(image_path)
      output = network_output(image, superpoint_model, maskrcnn_model, gcn_model, configs)
      data = {'image':image, 'points': output[0], 'objects': output[1],
          'descs': output[2], 'keeps': output[3]}
      
      # data = filter_objects(data, target_labels)
      labels = data['objects']['labels']
      print(labels)
      draw_results(tpl_data, data, save_dir, image_name)
    

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

  show_object_matching(configs)

if __name__ == "__main__":
  main()
