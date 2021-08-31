#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')   
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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from datasets.utils.build_data import coco_loader
from datasets.utils import pipeline as pp
from model.build_model import build_maskrcnn, build_gcn
from datasets.utils.preprocess import warp_batch_data, match_points_clusters
from model.graph_models.descriptor_loss import DescriptorLoss
from model.build_model import build_superpoint_model
from model.inference import superpoint_inference
from model.backbone.fcn import VGGNet
from model.superpoint.vgg_like import VggLike

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  save_dir = configs['save_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  data_aug_config = data_config['augmentation']
  # train_data_name = data_config['TRAIN']
  train_data_name = data_config['VAL']
  ## superpoint model config
  detection_threshold = configs['model']['superpoint']['detection_threshold']
  ## graph model config
  gcn_config = configs['model']['gcn']
  batch_szie = gcn_config['train']['batch_szie']
  epochs = gcn_config['train']['epochs']
  lr = gcn_config['train']['lr']
  momentum = gcn_config['train']['momentum']
  w_decay = gcn_config['train']['w_decay']
  milestones = gcn_config['train']['milestones']
  gamma = gcn_config['train']['gamma']
  checkpoint = gcn_config['train']['checkpoint']
  lambda_d = gcn_config['train']['lambda_d']
  weight_lambda = gcn_config['train']['weight_lambda']
  ## others
  configs['num_gpu'] = [0]
  configs['public_model'] = 0

  # data
  data_loader = coco_loader(data_root=data_root, name=train_data_name, config=data_config, 
      batch_size=batch_szie, remove_images_without_annotations=True)

  # model 
  superpoint_model = build_superpoint_model(configs, requires_grad=False)
  superpoint_model.eval()

  gcn_model = build_gcn(configs)
  gcn_model.train()

  # optimizer
  optimizer = optim.RMSprop(gcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

  # loss
  criterion = DescriptorLoss(gcn_config)
  
  sum_iter = 0
  for _ in range(epochs):
    for _, batch in enumerate(data_loader):
      optimizer.zero_grad()
      original_images = batch['image']
      original_sizes = [list(img.shape[-2:]) for img in original_images]
      points_output, maskrcnn_targets, _ = superpoint_inference(
          superpoint_model, batch, use_gpu, 1, data_config, detection_threshold, save_dir=None)

      warped_batch = warp_batch_data(batch, data_config)
      warped_points_output, warped_maskrcnn_targets, _ = superpoint_inference(
          superpoint_model, warped_batch, use_gpu, 1, data_config, detection_threshold, save_dir=None)
      
      masks = maskrcnn_targets['masks']
      warped_masks = warped_maskrcnn_targets['masks']
      if 'gcn_mask' in data_aug_config:
        gcn_aug = data_aug_config['gcn_mask']
        if gcn_aug['enable']:
          masks = pp.mask_augmentation(masks, gcn_aug)
          masks = torch.tensor(masks)

      batch_points, batch_descs, connections = match_points_clusters(points_output, masks, 
          warped_points_output, warped_masks)

      if len(connections) < 2:
        print("no object")
        continue
      
      batch_points = [points.cuda() for points in batch_points]
      batch_descs = [descs.cuda() for descs in batch_descs]
      batch_object_descs, locations = gcn_model(batch_points, batch_descs)
      connections = torch.stack(connections).cuda()

      # descriptor loss
      ploss, nloss = criterion(batch_object_descs, connections)

      # location loss
      locations_mean_loss = locations.mean()
      location_sum = torch.sum(locations, 0) 
      norm_locations_sum = torch.nn.functional.normalize(location_sum, p=2, dim=-1)
      # locations_norm_loss = 1 - norm_locations_sum.mean()
      zero = torch.tensor(0.0, dtype=norm_locations_sum.dtype, device=norm_locations_sum.device)
      locations_norm_loss = torch.max(zero, 0.1 - norm_locations_sum.mean())

      loss = ploss * lambda_d + nloss + locations_mean_loss * weight_lambda[0] + locations_norm_loss * weight_lambda[1]

      loss.backward()
      optimizer.step()
      scheduler.step()
      sum_iter = sum_iter + 1

      if sum_iter%1 == 0:
        print("sum_iter = {}, loss = {}".format(sum_iter, loss.item()))        
        print("ploss = {}, nloss = {}, locations_mean_loss = {}, locations_norm_loss = {}".format(
            ploss.item(), nloss.item(), locations_mean_loss.item(), locations_norm_loss.item()))        

      if sum_iter % checkpoint == 0:
        model_saving_path = os.path.join(save_dir, "gcn_model_{}.pth".format(sum_iter))
        torch.save(gcn_model.state_dict(), model_saving_path)
        print("saving model to {}".format(model_saving_path))


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
      "-sm", "--superpoint_model_path",
      dest = "superpoint_model_path",
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
  configs['superpoint_model_path'] = args.superpoint_model_path
  configs['graph_model_path'] = args.graph_model_path

  train(configs)

if __name__ == "__main__":
    main()