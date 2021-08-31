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

from datasets.utils.pipeline import makedir
from datasets.kitti.kitti_odomery import KittiOdometry
from datasets.utils import postprocess as post

from experiments.object_tracking.object_tracking import update_normal_size, network_output, calculate_pr_curve, box_iou
from experiments.utils.utils import save_tracking_results, plot_pr_curves, plot_tracking_details, get_pr_curve_area


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def filter_objects(descs, objects, target_labels):
  new_descs = []
  new_objects = {}
  for k in objects.keys():
    new_objects[k] = []

  for i in range(len(descs)):
    label = objects['labels'][i]
    score = objects['scores'][i]
    if label in target_labels and score > 0.5:
      new_descs.append(descs[i])
      for k in objects.keys():
        new_objects[k].append(objects[k][i])

  if len(new_descs) < 1:
    return None, None

  new_descs = np.vstack(new_descs)
  for k in objects.keys():
    new_objects[k] = np.vstack(new_objects[k])
  
  return new_descs, new_objects

def relocalization_offline(configs):

  # read configs
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  net_output_dir = configs['net_output_dir']
  dataset_name = configs['data']['name']

  # data
  seqs = ['00', '05', '06']
  similarity_thr = 0.88
  interval = 100
  pr_curves_list = []
  datasets_results = {}
  for seq in seqs:
    # dataset
    dataset = KittiOdometry(data_root, seq)
    dis_thr = dataset.dis_thr
    angle_thr = dataset.angle_thr
    interval = dataset.interval

    seq_net_output_dir = os.path.join(net_output_dir, seq)
    descs_list = []
    image_name_list = []
    image_indexes = []
    predict_loops = {}
    for data in dataset:
      image = data['image']
      image_name = data['image_name']
      gt = dataset.get_label(image_name)
      print(image_name)

      image_net_output_dir = os.path.join(seq_net_output_dir, image_name)
      # load points
      points = []
      points_dir = os.path.join(image_net_output_dir, "points")
      if not os.path.exists(points_dir):
        continue
      points_file_names = os.listdir(points_dir)
      for points_file_name in points_file_names:
        p_path = os.path.join(points_dir, points_file_name)
        p = np.load(p_path, allow_pickle=True)
        points.append(p)

      # load descs
      descs_path = os.path.join(image_net_output_dir, "descs.npy")
      descs = np.load(descs_path)

      # load objects
      objects = {}
      objects_dir = os.path.join(image_net_output_dir, "objects")
      objects_file_names = os.listdir(objects_dir)
      for objects_file_name in objects_file_names:
        key = objects_file_name.split('.')[0]
        value_path = os.path.join(objects_dir, objects_file_name)
        objects[key] = np.load(value_path)
    

      target_labels = [3, 8]
      descs, objects = filter_objects(descs, objects, target_labels)
      if descs is None:
        continue

      if gt['index'] > interval and len(descs_list) > 0:
        # find loop
        scores = []
        match_images = []
        for descs_i, image_name_i, image_index_i in zip(descs_list, image_name_list, image_indexes):
          if gt['index'] - image_index_i < interval:
            break
          
          descs_similarity = descs.dot(descs_i.T)  # m * n
          matches = descs_similarity > similarity_thr
          matches = matches * descs_similarity
          matches = np.max(matches, 1)
          # decide to match
          good_match = 0
          score = np.sum(matches)

          m, n = descs_similarity.shape
          num_diff = m - n
          num_diff = num_diff if num_diff > 0 else (-num_diff)
          score = score - num_diff * 0

          scores.append(score)
          match_images.append(image_name_i)

        predict_loops[image_name] = {'match_images': match_images, 'scores': scores}
        
      descs_list.append(descs)
      image_name_list.append(image_name)
      image_indexes.append(gt['index'])

    # find groundtruth
    loop_gt, num_loop_gt = dataset.get_loop_gt()
 
    # recall
    topk_k = [i for i in range(1, 21)]
    recalls = {}
    for k in topk_k:
      pred_loop = 0
      for image_name in loop_gt.keys():
        if loop_gt[image_name]:
          if image_name not in predict_loops.keys():
            continue

          predict_loop = predict_loops[image_name]
          scores, match_images = predict_loop['scores'], predict_loop['match_images']
          scores = torch.tensor(scores)
          k_loop_images = []
          k_loop_scores = []
          if k > len(scores):
            k_loop_images = match_images
          else:
            _, indices = scores.topk(k)
            indices = indices.numpy().tolist()
            for idx in indices:
              k_loop_images.append(match_images[idx])

          # if correct image in k_loop_images
          for k_image_name in k_loop_images:
            gt1 = dataset.get_label(image_name)
            gt2 = dataset.get_label(k_image_name)

            idx1, p1, R1 = gt1['index'], gt1['position'], gt1['rotation']
            idx2, p2, R2 = gt2['index'], gt2['position'], gt2['rotation']
            
            dp = np.linalg.norm((p1-p2))
            dR = R1.dot(R2.T)
            dr, _ = cv2.Rodrigues(dR)
            d_angle = np.linalg.norm(dr)
            d_idx = idx2 - idx1 
            d_idx = d_idx if d_idx > 0 else (-d_idx)
            if (d_idx > interval) and (dp < dis_thr) and (d_angle < angle_thr):
              pred_loop += 1
              break
      print("k = {}, pred_loop = {}, num_loop_gt = {}".format(k, pred_loop, num_loop_gt))
      recall = float(pred_loop) / num_loop_gt if num_loop_gt > 0 else 1
      recalls[k] = recall
    datasets_results[seq] = recalls


  file_name = "kitti_odometry.yaml"
  file_path = os.path.join(save_dir, file_name)
  fp = open(file_path, 'w')
  fp.write(yaml.dump(datasets_results))



def main():
  parser = argparse.ArgumentParser(description="show match")
  parser.add_argument(
      "-c", "--config_file",
      dest = "config_file",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-s", "--save_dir",
      dest = "save_dir",
      type = str, 
      default = "" 
  )
  parser.add_argument(
      "-n", "--net_output_dir",
      dest = "net_output_dir",
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
  configs['net_output_dir'] = args.net_output_dir
  configs['save_dir'] = args.save_dir

  relocalization_offline(configs)

if __name__ == "__main__":
  main()
