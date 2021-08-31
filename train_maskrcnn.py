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

from model.mask_rcnn.mask_rcnn import MaskRCNN
from datasets.utils.build_data import coco_loader
from datasets.utils.preprocess import preprocess_maskrcnn_train_data
from model.build_model import build_maskrcnn
  
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

def train(configs):
  # read configs
  ## command line config
  use_gpu = configs['use_gpu']
  model_dir = configs['model_dir']
  data_root = configs['data_root']
  ## data cofig
  data_config = configs['data']
  train_data_name = data_config['TRAIN']
  ## model config
  model_config = configs['model']['maskrcnn']
  train_batch_size = model_config['batch_size']
  epochs = model_config['epochs']
  lr = model_config['lr']
  momentum = model_config['momentum']
  w_decay = model_config['w_decay']
  milestones = model_config['milestones']
  gamma = model_config['gamma']
  checkpoint = model_config['checkpoint']
  ## others
  configs['num_gpu'] = [0, 1]

  # data
  train_loader = coco_loader(
      data_root=data_root, name=train_data_name, config=data_config, batch_size=train_batch_size, 
      remove_images_without_annotations=True)

  # model 
  model = build_maskrcnn(configs)

  model.train()
  
  # optimizer
  optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

  sum_iter = 0
  for _ in range(epochs):
    for iter, batch in enumerate(train_loader):
      optimizer.zero_grad()
      images, sizes, maskrcnn_targets  = preprocess_maskrcnn_train_data(batch, use_gpu, data_config)
      result = model(images, sizes, maskrcnn_targets) 

      losses_dict = result[0]
      losses_dict_print = {}
      for k in losses_dict:
        losses_dict[k] = torch.sum(losses_dict[k])
        losses_dict_print[k] = losses_dict[k].cpu().item()

      losses = [losses_dict[k] for k in losses_dict.keys()]
      losses = sum(losses)
      losses.backward()
      optimizer.step()

      if iter%10 == 0:
        print("sum_iter = {}, loss = {}".format(sum_iter, losses.item()))
        print("loss_dict = {}".format(losses_dict_print))

      if sum_iter % checkpoint == 0:
        model_saving_path = os.path.join(model_dir, "maskrcnn_iter{}.pth".format(sum_iter))
        torch.save(model.state_dict(), model_saving_path)
        print("saving model to {}".format(model_saving_path))

      scheduler.step()  
      sum_iter += 1

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
  parser.add_argument(
      "-p", "--public_model",
      dest = "public_model",
      type = int, 
      default = 0 
  )
  args = parser.parse_args()

  config_file = args.config_file
  f = open(config_file, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)
  configs['use_gpu'] = args.gpu
  configs['model_dir'] = args.save_dir
  configs['data_root'] = args.data_root
  configs['maskrcnn_model_path'] = args.pretrained_model_path
  configs['public_model'] = args.public_model

  train(configs)

if __name__ == "__main__":
    main()


    
