#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os
import copy
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torchvision import transforms

from utils.tools import tensor_to_numpy
from datasets.utils.preprocess import preprocess_validation_data
from datasets.utils import postprocess as post
from utils.tools import tensor_to_numpy
from datasets.utils.pipeline import makedir

def detection_inference(maskrcnn_model, superpoint_model, batch, use_gpu, gaussian_radius, detection_threshold,
    data_config, save_dir=None):
  with torch.no_grad():
    original_images = batch['image']
    original_images = [tensor_to_numpy(img.clone()) for img in original_images]

    # preprocess
    images, sizes, maskrcnn_targets, superpoint_targets = preprocess_validation_data(batch, 
        use_gpu, gaussian_radius, data_config)
    original_sizes = sizes['original_sizes']
    new_sizes = sizes['new_sizes']

    # model inference
    _, detections = maskrcnn_model(images, sizes) 
    points_output = superpoint_model(images) 

    # postprocess
    detections, points_output = post.postprocess(new_sizes, original_sizes, detection_threshold,
       detections, points_output)

    # save results
    if save_dir is not None:
      image_names = batch['image_name']
      results = post.save_detection_results(original_images, image_names, save_dir, detections,
         None, points_output, True, True)

  return points_output, detections, maskrcnn_targets


def maskrcnn_inference(model, batch, use_gpu, gaussian_radius, data_config, save_dir=None):
  with torch.no_grad():
    original_images = batch['image']
    original_images = [tensor_to_numpy(img.clone()) for img in original_images]

    # preprocess
    images, sizes, maskrcnn_targets, _ = preprocess_validation_data(batch, use_gpu, gaussian_radius, data_config)
    original_sizes = sizes['original_sizes']
    new_sizes = sizes['new_sizes']

    # model inference
    _, detections = model(images, sizes) 

    # postprocess
    detections, _ = post.postprocess(new_sizes, original_sizes, detections=detections)

    # save results
    if save_dir is not None:
      image_names = batch['image_name']
      results = post.save_detection_results(original_images, image_names, save_dir, detections, None, None, True, False)

  return detections, maskrcnn_targets


def superpoint_inference(model, batch, use_gpu, gaussian_radius, data_config, detection_threshold, save_dir=None):
  with torch.no_grad():
    original_images = batch['image']
    original_images = [tensor_to_numpy(img.clone()) for img in original_images]

    # preprocess
    images, sizes, maskrcnn_targets, superpoint_targets = preprocess_validation_data(batch, use_gpu, gaussian_radius, data_config)
    original_sizes = sizes['original_sizes']
    new_sizes = sizes['new_sizes']

    # model inference
    points_output = model(images) 

    # postprocess
    _, points_output = post.postprocess(new_sizes, original_sizes, detection_threshold, None, points_output)

    # save gt 
    if save_dir is not None:
      print("save_dir = {}".format(save_dir))
      save_dir_list = [os.path.join(save_dir, image_name) for image_name in batch['image_name']]
      for d in save_dir_list:
        makedir(d)
      images = copy.deepcopy(original_images)
      images = post.overlay_points(images, points_output)
      post.save_images(images, save_dir_list, "points")

  return points_output, maskrcnn_targets, superpoint_targets