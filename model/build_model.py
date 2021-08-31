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
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model.backbone.resnet_fpn import resnet_fpn_backbone
from model.backbone.fcn import VGGNet
from model.mask_rcnn.mask_rcnn import MaskRCNN
from model.superpoint.vgg_like import VggLike
from model.superpoint.superpoint_public_model import SuperPointNet
from model.graph_models.object_descriptor import ObjectDescriptor

def build_maskrcnn(configs):
  ## command line config
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0) and configs['use_gpu']
  pretrained_model_path = configs['maskrcnn_model_path']
  public_model = configs['public_model']
  ## data cofig
  nclass = configs['data']['nclass']
  ## mask_rcnn config
  maskrcnn_model_config = configs['model']['maskrcnn']
  backbone_type = maskrcnn_model_config['backbone_type']
  image_mean = maskrcnn_model_config['image_mean']
  image_std = maskrcnn_model_config['image_std']
  trainable_layers = maskrcnn_model_config['trainable_layers']

  # model 
  # backbone = ResNetFPN(pretrained_type)
  backbone = resnet_fpn_backbone(backbone_type, False, trainable_layers=trainable_layers)
  model = MaskRCNN(backbone, nclass, image_mean=image_mean, image_std=image_std)

  if pretrained_model_path != "" and public_model:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    remove_dict = ['roi_heads.box_predictor.cls_score.weight',
                   'roi_heads.box_predictor.cls_score.bias',
                   'roi_heads.box_predictor.bbox_pred.weight',
                   'roi_heads.box_predictor.bbox_pred.bias',
                   'roi_heads.mask_predictor.mask_fcn_logits.weight',
                   'roi_heads.mask_predictor.mask_fcn_logits.bias']
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if ((k in model_dict) and (k not in remove_dict))}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("load model from {}".format(pretrained_model_path))
    print("load parameters : {}".format(pretrained_dict.keys()))
    

  if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "" and (not public_model):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("load model from {}".format(pretrained_model_path))

  return model

def build_superpoint_model(configs, requires_grad=True):
  ## command line config
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0) and configs['use_gpu']
  pretrained_model_path = configs['superpoint_model_path']

  vgg_model = VGGNet(requires_grad=requires_grad)
  model = VggLike(vgg_model)
  
  # model = SuperPointNet()
  # if pretrained_model_path != "":
  #   model_dict = model.state_dict()
  #   pretrained_dict = torch.load(pretrained_model_path)
  #   model_dict.update(pretrained_dict)
  #   model.load_state_dict(model_dict)
  #   print("load model from {}".format(pretrained_model_path))


  if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("load model from {}".format(pretrained_model_path))
  
  return model

def build_gcn(configs):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  gcn_config = configs['model']['gcn']
  pretrained_model_path = configs['graph_model_path']

  model = ObjectDescriptor(gcn_config)

  if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    if use_gpu:  
      model.load_state_dict(torch.load(pretrained_model_path))
    else:
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model