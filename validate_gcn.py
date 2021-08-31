# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import argparse
import yaml

import torch
import torch.distributed as dist

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from datasets.utils.build_data import coco_loader
from model.build_model import build_maskrcnn, build_gcn
from datasets.utils.preprocess import warp_batch_data, match_points_clusters
from validate import maskrcnn_inference
from model.graph_models.descriptor_loss import DescriptorLoss


def calculate_f1(maskrcnn_model, gcn_model, loader, configs):
  with torch.no_grad():
    maskrcnn_model.eval()
    gcn_model.eval()

    ## data cofig
    data_config = configs['data']
    ## superpoint model config
    superpoint_model_config = configs['model']['superpoint']
    detection_threshold = superpoint_model_config['eval']['detection_threshold']

    precisions, recalls, weights = [], [], []

    for iter, batch in enumerate(loader): 
      optimizer.zero_grad()
      original_images = batch['image']
      original_sizes = [list(img.shape[-2:]) for img in original_images]
      _, points_output, maskrcnn_targets, _ = maskrcnn_inference(
          maskrcnn_model, batch, use_gpu, 1, data_config, detection_threshold)

      warped_batch = warp_batch_data(batch, data_config)
      _, warped_points_output, warped_maskrcnn_targets, _ = maskrcnn_inference(
          maskrcnn_model, warped_batch, use_gpu, 1, data_config, detection_threshold)
      
      batch_points, batch_descs, connections = match_points_clusters(points_output, maskrcnn_targets['masks'], 
          warped_points_output, warped_maskrcnn_targets['masks'])

      if len(connections) < 2:
        print("no object")
        continue
      
      batch_points = [points.cuda() for points in batch_points]
      batch_descs = [descs.cuda() for descs in batch_descs]
      batch_object_descs = gcn_model(batch_points, batch_descs)
      connections = torch.stack(connections).cuda()

      distances = torch.einsum('nd,dm->nm', descs, descs.t())  # N * N
      good_matchs = (distances > dist_thr).float() 
      num_correct_matches = torch.sum(good_matchs * connections)
      num_connections = torch.sum(connections)

      recall = num_correct_matches / num_connections
      precision = num_correct_matches / torch.sum(good_matchs)

      recalls.append(recall)
      precisions.append(precision)
      weights.append(num_connections)

    if(len(weights) == 0):
      return 0., 0., 0.

    recalls = torch.tensor(recalls)
    precisions = torch.tensor(precisions)
    weights = torch.tensor(weights)

    total_number = torch.sum(weights)
    aver_recall = torch.sum(recalls * weights) / total_number
    aver_precision = torch.sum(precisions * weights) / total_number
    aver_f1 = 0. if (aver_recall + aver_precision) = 0 else aver_recall * aver_precision / (aver_recall + aver_precision)

    return aver_recall, aver_precision, aver_f1


def validate(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  validation_data_name = data_config['VAL']
  ## superpoint model config
  superpoint_model_config = configs['model']['superpoint']
  detection_threshold = superpoint_model_config['eval']['detection_threshold']
  ## graph model config
  gcn_config = configs['model']['gcn']
  batch_szie = gcn_config['train']['batch_szie']
  ## others
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # data
  data_loader = coco_loader(data_root=data_root, name=validation_data_name, config=data_config, 
      batch_size=batch_szie, remove_images_without_annotations=True)

  # model 
  maskrcnn_model = build_maskrcnn(configs)
  gcn_model = build_gcn(configs)

  recall, precision, f1 = calculate_f1(maskrcnn_model, gcn_model, data_loader, configs)
  print("recall = {}, precision = {}, f1 = {}".format(recall, precision, f1))


def main():
  parser = argparse.ArgumentParser(description="Validation")
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
      "-gm", "--graph_model_path",
      dest = "graph_model_path",
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
  configs['pretrained_model_path'] = args.maskrcnn_model_path
  configs['graph_model_path'] = args.graph_model_path

  validate(configs)

if __name__ == "__main__":
    main()
