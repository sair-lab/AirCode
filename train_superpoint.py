# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import argparse
import yaml
import copy

import torch
import torch.distributed as dist

import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader

from model.build_model import build_superpoint_model
from model.superpoint.superpoint_loss import SuperPointLoss
from datasets.utils.build_data import coco_loader
from datasets.synthetic.synthetic import SyntheticDataset
from datasets.utils.batch_collator import BatchCollator
from datasets.utils.preprocess import preprocess_superpoint_train_data


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

def update_gaussian_radius(gaussian_radius, iter, gaussian_gamma, gaussian_milestones):
  r = gaussian_radius
  if r < 0:
    return 1, gaussian_radius
  
  for i in range(len(gaussian_milestones)):
    if iter > gaussian_milestones[i]:
      r = r * gaussian_gamma
    else:
      break
  
  r = int(r)
  if r < 2:
    gaussian_radius = -1
  return r, gaussian_radius

def train(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  model_dir = configs['model_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  dataset_name = data_config['name']
  ## superpoint model config
  superpoint_model_config = configs['model']['superpoint']
  train_batch_size = superpoint_model_config['train']['batch_size']
  epochs = superpoint_model_config['train']['epochs']
  lr = superpoint_model_config['train']['lr']
  momentum = superpoint_model_config['train']['momentum']
  w_decay = superpoint_model_config['train']['w_decay']
  milestones = superpoint_model_config['train']['milestones']
  gamma = superpoint_model_config['train']['gamma']
  gaussian_region = superpoint_model_config['train']['gaussian_region']
  gaussian_radius = gaussian_region['radius']
  gaussian_gamma = gaussian_region['gamma']
  gaussian_milestones = gaussian_region['milestones']
  train_batch_size = superpoint_model_config['train']['batch_size']
  checkpoint = superpoint_model_config['train']['checkpoint']
  ## others
  configs['num_gpu'] = [0, 1]
 
  # data
  if 'coco' in dataset_name:
    train_data_name = data_config['TRAIN']
    train_loader = coco_loader(
        data_root=data_root, name=train_data_name, config=data_config, batch_size=train_batch_size, 
        remove_images_without_annotations=True)
  elif 'synthetic' in dataset_name:
    train_dataset = SyntheticDataset(data_root=data_root, use_for='training')
    sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=train_batch_size, drop_last=True)
    collator = BatchCollator()
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=collator, num_workers=8)

  # model 
  model = build_superpoint_model(configs)
  model.train()
  
  # optimizer
  optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

  # loss
  criterion = SuperPointLoss(config=superpoint_model_config)

  sum_iter = 0
  r = gaussian_radius
  for _ in range(epochs):
    for iter, batch in enumerate(train_loader):
      optimizer.zero_grad()
      batch = preprocess_superpoint_train_data(batch, use_gpu, r, data_config)

      if use_gpu:
        for key in batch:
          if key == 'image_name':
            continue
          batch[key] = batch[key].cuda()

      outputs = model(batch['image'])
      batch_outputs = {'outputs': outputs}
      if 'warped_image' in batch:
        warped_outputs = model(batch['warped_image'])
        batch_outputs['warped_outputs'] = warped_outputs

      loss, loss_dict = criterion(batch, batch_outputs)
      loss = loss / train_batch_size

      for k in loss_dict:
        loss_dict[k] = loss_dict[k].cpu().item() / train_batch_size

      loss.backward()
      optimizer.step()
    
      if iter%10 == 0:
        print("sum_iter = {}, gaussian_radius={}, loss = {}".format(sum_iter, r, loss.item()))
        
      sum_iter += 1
      r, gaussian_radius = update_gaussian_radius(gaussian_radius, sum_iter, gaussian_gamma, gaussian_milestones)
      scheduler.step()

      if sum_iter % checkpoint == 0:
        model_saving_path = os.path.join(model_dir, "superpoint_iter{}.pth".format(sum_iter))
        torch.save(model.state_dict(), model_saving_path)
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
  configs['model_dir'] = args.save_dir
  configs['data_root'] = args.data_root
  configs['superpoint_model_path'] = args.pretrained_model_path

  train(configs)

if __name__ == "__main__":
    main()


    
