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

from model.build_model import build_maskrcnn, build_gcn
from model.inference import maskrcnn_inference
from model.build_model import build_maskrcnn, build_gcn, build_superpoint_model
from model.inference import detection_inference
from model.backbone.fcn import VGGNet
from model.superpoint.vgg_like import VggLike
from datasets.utils.pipeline import makedir
from debug_tools.show_batch import show_batch, show_numpy
from utils.tools import tensor_to_numpy
from kornia.feature import match_nn
from datasets.utils import preprocess, postprocess
from datasets.kitti.kitti_tracking import KittiTracking
from datasets.otb.otb_tracking import OtbTracking
from datasets.vot.vot_tracking import VotTracking

from experiments.utils.utils import save_tracking_results, plot_pr_curves, plot_tracking_details, get_pr_curve_area


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def network_output(image, points_model, maskrcnn_model, gcn_model, configs):  
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
    good_boxes, good_masks, good_labels, good_scores = [], [], [], []
    for i in range(len(keeps)):
      if keeps[i].item():
        good_points.append(batch_points[i])
        good_descs.append(batch_descs[i])
        good_boxes.append(detections[0]['boxes'][i])
        good_masks.append(detections[0]['masks'][i])
        good_labels.append(detections[0]['labels'][i])
        good_scores.append(detections[0]['scores'][i])

    if len(good_points) == 0:
      return None, None, None

    batch_object_descs, _ = gcn_model(good_points, good_descs)

    good_boxes = torch.stack(good_boxes)
    good_masks = torch.stack(good_masks)

    good_labels = torch.stack(good_labels)
    good_scores = torch.stack(good_scores)
    detections = {'boxes':good_boxes, 'masks':good_masks, 'labels':good_labels, 'scores':good_scores}

    return good_points, detections, batch_object_descs


def update_normal_size(size):
  min_size, max_size = size
  min_size = (1 + (min_size - 1) // 32) * 32
  max_size = (1 + (max_size - 1) // 32) * 32
  return [min_size, max_size]
  

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def assign_descs(net_output, tracking_gt):
  '''
  assign object descriptor to tracking label
  '''

  net_output_boxes = net_output['objects']['boxes']

  boxes = torch.tensor(tracking_gt['box'])
  ious = box_iou(boxes, net_output_boxes)

  descs = []
  for iou in ious:
    value, index = iou.max(0)
    if value > 0.5:
      desc = net_output['descs'][index]
    else:
      desc = None
    descs.append(desc)
  tracking_gt['descs'] = descs
  
  return tracking_gt

def sample_objects(frame_batch_info):
  '''
  remove objects without descriptors
  '''
  descs = frame_batch_info['descs']
  new_frame_batch_info = {}
  for k in frame_batch_info.keys():
    data_list = [frame_batch_info[k][i] for i in range(len(descs)) if descs[i] is not None]
    new_frame_batch_info[k] = data_list
  
  return new_frame_batch_info


def match_objects(frame_batch_info, last_frame_batch_info):
  '''
  calculate gt_matrix, match_matrix
  '''

  # generate groundtruth pair matrix
  track_ids = frame_batch_info['track_id']
  last_track_ids = last_frame_batch_info['track_id']

  N, M = len(track_ids), len(last_track_ids)

  if N == 0 or M == 0:
    return None, None
  gt_matrix = torch.zeros(N, M)
  for i in range(N):
    track_id = track_ids[i]
    if track_id in last_track_ids:
      j = last_track_ids.index(track_id)
      gt_matrix[i, j] = 1.0

  # generate matching matrix
  descs = torch.stack(frame_batch_info['descs'])
  last_descs = torch.stack(last_frame_batch_info['descs'])
  match_matrix = torch.einsum('nd,dm->nm', descs, last_descs.t())  # N * M
  
  return gt_matrix, match_matrix


def calculate_pr_curve(gts, matches):
  # calculate p, r
  thrs = [float(i)/50 for i in range(51)]
  pr_curve = []
  for thr in thrs:
    pr_numbers = []
    for gt_matrix, match_matrix in zip(gts, matches):
      match_matrix = (match_matrix.cpu() > thr).float()
      tp = torch.sum(gt_matrix.cpu() * match_matrix).item()
      match_num = torch.sum(match_matrix).item()
      gt_num = torch.sum(gt_matrix).item()
     
      pr_number = [tp, match_num, gt_num]
      pr_numbers.append(pr_number)

    pr_numbers = torch.tensor(pr_numbers)
    pr_numbers = torch.sum(pr_numbers, 0)

    TP, MatchNum, GTNum = pr_numbers.cpu().numpy().tolist()

    precision = TP / MatchNum if MatchNum > 0 else 1
    recall = TP / GTNum if GTNum > 0 else 1
    pr_curve.append([precision, recall])

  return pr_curve


def calculate_pr_curves(frame_batch_info_list, intervals):
  pr_curves = {}
  for interval in intervals:
    gts, matches = [], []
    last_frame_batch_info = None
    for i in range(len(frame_batch_info_list)):
      if (i % interval) != 0:
        continue
      
      frame_batch_info = frame_batch_info_list[i]
      if last_frame_batch_info is not None and frame_batch_info is not None:
        gt_matrix, match_matrix = match_objects(frame_batch_info, last_frame_batch_info)
        if gt_matrix is not None and match_matrix is not None:
          gts.append(gt_matrix)
          matches.append(match_matrix)

      last_frame_batch_info = frame_batch_info

    pr_curve = calculate_pr_curve(gts, matches)
    pr_curves[interval] = pr_curve
  return pr_curves


def show_object_tracking(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  model_dir = configs['model_dir']
  use_gpu = configs['use_gpu']
  superpoint_model_config = configs['model']['superpoint']
  detection_thr = superpoint_model_config['detection_threshold']
  img_new_size = configs['img_new_size']
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
    
    frame_batch_info_list = []
    for data in dataset:
      image = data['image']
      image_name = data['image_name']
      print(image_name)

      net_output = network_output(image, superpoint_model, maskrcnn_model, gcn_model, configs)
      net_output = {'points': net_output[0], 'objects': net_output[1], 'descs': net_output[2]}

      tracking_gt = dataset.get_label(image_name)
      if tracking_gt is None or net_output['points'] is None:
        frame_batch_info_list.append(None)
        continue

      frame_batch_info = assign_descs(net_output, tracking_gt)
      frame_batch_info = sample_objects(frame_batch_info)
      frame_batch_info_list.append(frame_batch_info)

    pr_curves = calculate_pr_curves(frame_batch_info_list, intervals)
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

  plot_pr_curves(new_pr_curves, "kitti_tracking", save_dir)

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
  configs['img_new_size'] = [480, 640]

  show_object_tracking(configs)

if __name__ == "__main__":
  main()
