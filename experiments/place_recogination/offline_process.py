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
  similarity_thr = 0.95
  results = {}
  for seq in seqs:
    # dataset
    dataset = KittiOdometry(data_root, seq)
    dis_thr = dataset.dis_thr
    angle_thr = dataset.angle_thr
    interval = dataset.interval
    loop_gt, num_loop_gt = dataset.get_loop_gt()

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
      #   objects[key] = torch.tensor(objects[key])
 
      # original_images = [image]
      # image_names = [image_name]
      # detections = [objects]
      # results = post.save_detection_results(original_images, image_names, save_dir, detections,
      #    None, None, True, False)
    

      target_labels = [3, 8]
      descs, objects = filter_objects(descs, objects, target_labels)
      if descs is None:
        continue

      if gt['index'] > interval and len(descs_list) > 0:
        # find loop
        max_score = 0
        match_image = ""
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
          num_nonzero = np.sum((matches > 0))
          mean_match = np.mean(matches)
          if score > 0 and mean_match > 0.3:
            good_match = 1

          m, n = descs_similarity.shape
          if m - n > 2 or n - m > 2:
            good_match = 0

          if good_match and score > max_score:
            max_score = score
            match_image = image_name_i
        predict_loops[image_name] = {'image_name': match_image, 'score': max_score}
        
      descs_list.append(descs)
      image_name_list.append(image_name)
      image_indexes.append(gt['index'])

    # calculate pr
    # calculate prediction
    num_loop_prediction, num_correct_prediction = 0, 0
    for image_name in predict_loops.keys():
      predict_loop = predict_loops[image_name]
      if predict_loop['score'] > 0:
        num_loop_prediction += 1
        loop_image_name = predict_loop['image_name']
        gt1 = dataset.get_label(image_name)
        gt2 = dataset.get_label(loop_image_name)

        idx1, p1, R1 = gt1['index'], gt1['position'], gt1['rotation']
        idx2, p2, R2 = gt2['index'], gt2['position'], gt2['rotation']
        
        dp = np.linalg.norm((p1-p2))
        dR = R1.dot(R2.T)
        dr, _ = cv2.Rodrigues(dR)
        d_angle = np.linalg.norm(dr)
        d_idx = idx2 - idx1 
        d_idx = d_idx if d_idx > 0 else (-d_idx)
        if (d_idx > interval) and (dp < dis_thr) and (d_angle < angle_thr):
          num_correct_prediction += 1
        # else:
        #   print("img1 = {}, img2 = {}".format(image_name, loop_image_name))

    precision = float(num_correct_prediction) / num_loop_prediction if num_loop_prediction > 0 else 1
    recall = float(num_correct_prediction) / num_loop_gt if num_loop_gt > 0 else 1

    results[seq] = {'precision': precision, 'recall': recall}
  
  print(results)
  file_name = "kitti_odometry_pr.yaml"
  file_path = os.path.join(save_dir, file_name)
  fp = open(file_path, 'w')
  fp.write(yaml.dump(results))


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
