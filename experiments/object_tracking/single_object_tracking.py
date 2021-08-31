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
from datasets.kitti.kitti_tracking import KittiTracking
from datasets.otb.otb_tracking import OtbTracking
from datasets.vot.vot_tracking import VotTracking

from experiments.object_tracking.object_tracking import update_normal_size, network_output, calculate_pr_curve, box_iou
from experiments.utils.utils import save_tracking_results, plot_pr_curves, plot_tracking_details, get_pr_curve_area


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def reorder_descs(net_output, tracking_gt):
  '''
  reorder the descs, the first is the desc of tracked object
  '''
  net_output_boxes = net_output['objects']['boxes']  # N * 4
  if len(net_output_boxes) < 1 : 
    return None

  boxes = torch.tensor(tracking_gt['box'])  # 4
  boxes = boxes.unsqueeze(0)  # 1 * 4
  ious = box_iou(boxes, net_output_boxes) # 1 * N
  ious = ious.squeeze(0)

  value, index = ious.max(0)
  value, index = value.item(), index.item()

  if value > 0.5:
    descs = net_output['descs']
    order = [i for i in range(len(descs))]
    order[0] = index
    order[index] = 0

    descs = descs[order]
  else:
    descs = None

  return descs


def match_objects(object_descs, last_object_descs):
  '''
  calculate gt_matrix, match_matrix
  '''
  if object_descs is None or last_object_descs is None:
    return None, None

  # generate groundtruth pair matrix
  def get_gt_and_match(descs1, descs2):
    N = len(descs2)
    gt_matrix = torch.zeros(N)
    gt_matrix[0] = 1.0
 
    tracked_desc = descs1[0].unsqueeze(0) # 1 * D
    match_matrix = torch.einsum('nd,dm->nm', tracked_desc, descs2.t())  # 1 * M
    return gt_matrix, match_matrix

  gt1, match1 = get_gt_and_match(last_object_descs, object_descs)
  gt2, match2 = get_gt_and_match(object_descs, last_object_descs)
  
  return [gt1, gt2], [match1, match2]


def calculate_pr_curves(object_descs_list, intervals):
  pr_curves = {}
  for interval in intervals:
    gts, matches = [], []
    last_object_descs = None
    for i in range(len(object_descs_list)):
      if (i % interval) != 0:
        continue
      
      object_descs = object_descs_list[i]
      if last_object_descs is not None and object_descs is not None:
        gt_matrix, match_matrix = match_objects(object_descs, last_object_descs)
        if gt_matrix is not None and match_matrix is not None:
          gts += gt_matrix
          matches += match_matrix

      last_object_descs = object_descs

    pr_curve = calculate_pr_curve(gts, matches)
    pr_curves[interval] = pr_curve
  return pr_curves


def single_object_tracking(configs):

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

  intervals = [1, 2, 3, 5, 10]
  seqs = {'kitti':['0002', '0003', '0006', '0010'],
          'otb': ['BlurBody', 'BlurCar2', 'Human2', 'Human7', 'Liquor'],
          'vot': ['bluecar', 'bus6', 'humans_corridor_occ_2_A']}
  datasets = {'kitti':KittiTracking, 'otb':OtbTracking, 'vot':VotTracking}
  DATASET = datasets[dataset_name]
  SEQNAMES = seqs[dataset_name]
  # SEQNAMES = [seqs[dataset_name][0]]

  pr_curves_list = []
  for seq in SEQNAMES:
    dataset = DATASET(data_root, seq)

    image_size = dataset.image_size()
    configs['data']['normal_size'] = update_normal_size(image_size)
    
    object_descs_list = []
    for data in dataset:
      image = data['image']
      image_name = data['image_name']
      print(image_name)

      net_output = network_output(image, superpoint_model, maskrcnn_model, gcn_model, configs)
      net_output = {'points': net_output[0], 'objects': net_output[1], 'descs': net_output[2]}

      tracking_gt = dataset.get_label(image_name)
      if tracking_gt is None or net_output['points'] is None:
        object_descs_list.append(None)
        continue

      object_descs = reorder_descs(net_output, tracking_gt)
      object_descs_list.append(object_descs)

    pr_curves = calculate_pr_curves(object_descs_list, intervals)
    pr_curves_list.append(pr_curves)
    print(pr_curves)

  # plot
  new_pr_curves, areas = {}, {}
  for k in pr_curves_list[0].keys():
    pr_curve_list = [torch.tensor(pr_curves[k]) for pr_curves in pr_curves_list]
    pr_curve_list = torch.stack(pr_curve_list)  # N * 10 * 2
    new_pr_curve = torch.mean(pr_curve_list, 0)
    new_pr_curves[k] = new_pr_curve.cpu().numpy().tolist()
    areas[k] = get_pr_curve_area(new_pr_curves[k])
  
  plot_pr_curves(new_pr_curves, "otb_tracking", save_dir)

  # # save results to yaml
  results = {'dataset':dataset_name, 'model':"ours", 'areas': areas, 'pr_curves': new_pr_curves}
  save_tracking_results(results, save_dir)


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

  single_object_tracking(configs)

if __name__ == "__main__":
  main()
