#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os

import torch
import torch.distributed as dist
from torchvision import transforms
import copy

from datasets.utils import transforms as T
from datasets.utils import pipeline as pp
from utils.tools import tensor_to_numpy
from datasets.utils.homographies import warp_batch_images, warp_points

def unify_size(size_list):
  v, _ = size_list.max(0)

  max_H, max_W = v[0].item(), v[1].item()
  
  new_H = (1 + (max_H - 1) // 32) * 32
  new_W = (1 + (max_W - 1) // 32) * 32

  return (new_H, new_W)

def pad_images(image_list, new_size=None, to_stack=False):
  if new_size is None:
    image_sizes = [(img.shape[-2], img.shape[-1]) for img in image_list]
    image_sizes = torch.tensor(image_sizes)
    new_size = unify_size(image_sizes)
  
  new_images = []
  for i in range(len(image_list)):
    size = image_list[i].shape[-2:]
    padding_bottom = new_size[0] - size[0]
    padding_right = new_size[1] - size[1]
    new_images += [torch.nn.functional.pad(image_list[i], (0, padding_right, 0, padding_bottom))]
  
  if to_stack:
    new_images = torch.stack(new_images, 0)
  
  return new_images, new_size

def preprocess_maskrcnn_train_data(batch, use_gpu, config):
  
  min_szie, max_size = config['normal_size']

  # utar targets
  boxes, labels, masks = batch['boxes'], batch['labels'], batch['masks']

  # resize image and mask
  images = batch['image']
  new_images = []
  original_sizes = []
  new_sizes = []
  for i in range(len(images)):
    image = images[i]
    original_size = [image.shape[-2], image.shape[-1]]
    original_sizes.append(original_size)
    target = {'boxes': boxes[i], 'labels':labels[i], 'masks':masks[i]}
    image, target = T.resize(image, target, min_szie, max_size)
    new_size = [image.shape[-2], image.shape[-1]]
    new_sizes.append(new_size)
    new_images.append(image)
    boxes[i], labels[i], masks[i] = target['boxes'], target['labels'], target['masks']
  images = new_images

  # pad maskrcnn targets
  num_objects = [len(c) for c in labels]
  max_num = max(num_objects)
  for i in range(len(boxes)):
    padding_num = max_num - len(labels[i])
    boxes[i] = torch.nn.functional.pad(boxes[i], [0, 0, 0, padding_num])
    labels[i] = torch.nn.functional.pad(labels[i], [0, padding_num], value=-1)
    masks[i] = torch.nn.functional.pad(masks[i], [0, 0, 0, 0, 0, padding_num]) 
  
  # batch data
  images, new_size = pad_images(images, to_stack=True)
  masks, _ = pad_images(masks, new_size, True)

  original_sizes = torch.tensor(original_sizes)
  new_sizes = torch.tensor(new_sizes)

  maskrcnn_targets = {'boxes':boxes, 'labels':labels, 'masks':masks}
  sizes = {'original_sizes':original_sizes, 'new_sizes':new_sizes}

  return images, sizes, maskrcnn_targets


def preprocess_superpoint_train_data(batch, use_gpu, gaussian_radius, config):

  min_szie, max_size = config['normal_size']
  image_to_tensor = transforms.ToTensor()

  # resize image and mask
  images = batch['image']
  new_images = []
  original_sizes = []
  new_sizes = []
  for i in range(len(images)):
    image = images[i]
    original_size = [image.shape[-2], image.shape[-1]]
    original_sizes.append(original_size)
    image, _ = T.resize(image, None, min_szie, max_size)
    new_size = [image.shape[-2], image.shape[-1]]
    new_sizes.append(new_size)
    new_images.append(image)
  images = new_images

  # generate superpoint data
  points_list = batch['points']
  valid_masks, keypoint_maps, heatmaps = [], [], []
  warped_images, warped_valid_masks, warped_keypoint_maps, warped_heatmaps, Hs = [], [], [], [], []
  for i in range(len(images)):
    image = tensor_to_numpy(images[i])
    points = T.resize_keypoints(points_list[i], original_sizes[i], new_sizes[i]) 
    points = points.data.cpu().numpy()

    if config['augmentation']['photometric']['enable']:
      image, points = pp.photometric_augmentation(image, points, config['augmentation']['photometric'])

    valid_mask = pp.generate_valid_mask(image.shape[:2])
    keypoint_map = pp.generate_keypoint_map(image.shape[:2], points)
    heatmap = pp.generate_heatmap(image.shape[:2], points, gaussian_radius)
    
    valid_mask = torch.tensor(valid_mask)
    keypoint_map = torch.tensor(keypoint_map)
    keypoint_map = keypoint_map.unsqueeze(0)
    heatmap = torch.tensor(heatmap)
    heatmap = heatmap.unsqueeze(0)

    keypoint_maps.append(keypoint_map)
    valid_masks.append(valid_mask)
    heatmaps.append(heatmap)

    if config['warped_pair']['enable']:
      warped_image, warped_points, warped_valid_mask, H = pp.homographic_augmentation(image, points, config['warped_pair'])

      if config['augmentation']['photometric']['enable']:
        warped_image, warped_points = pp.photometric_augmentation(warped_image, warped_points, config['augmentation']['photometric'])

      warped_keypoint_map = pp.generate_keypoint_map(warped_image.shape[:2], warped_points)
      warped_heatmap = pp.generate_heatmap(warped_image.shape[:2], warped_points, gaussian_radius)

      warped_image = image_to_tensor(warped_image)
      warped_valid_mask = torch.tensor(warped_valid_mask)
      warped_keypoint_map = torch.tensor(warped_keypoint_map)
      warped_keypoint_map = warped_keypoint_map.unsqueeze(0)
      warped_heatmap = torch.tensor(warped_heatmap)
      warped_heatmap = warped_heatmap.unsqueeze(0)
      H = torch.tensor(H)
      H = H.unsqueeze(0)

      warped_images.append(warped_image)
      warped_keypoint_maps.append(warped_keypoint_map)
      warped_valid_masks.append(warped_valid_mask)
      warped_heatmaps.append(warped_heatmap)
      Hs.append(H)

  # batch data
  images, new_size = pad_images(images, to_stack=True)

  data_list = [valid_masks, keypoint_maps, heatmaps]
  for i in range(len(data_list)):
    data_list[i], _ = pad_images(data_list[i], new_size, True)
  valid_masks, keypoint_maps, heatmaps = data_list

  batch_data = {'image':images, 'valid_mask':valid_masks, 'kpt_map':keypoint_maps, 'ht':heatmaps}


  if config['warped_pair']['enable']: 
    Hs = torch.stack(Hs, 0)
    data_list = [warped_images, warped_valid_masks, warped_keypoint_maps, warped_heatmaps]
    for i in range(len(data_list)):
      data_list[i], _ = pad_images(data_list[i], new_size, True)
    warped_images, warped_valid_masks, warped_keypoint_maps, warped_heatmaps = data_list
    warped_batch_data = {'warped_image':warped_images, 'warped_valid_mask':warped_valid_masks, 
        'warped_keypoint_map':warped_keypoint_maps, 'warped_ht':warped_heatmaps, 'H':Hs} 
    batch_data.update(warped_batch_data)

  return batch_data


def preprocess_validation_data(batch, use_gpu, gaussian_radius, config):

  min_size, max_size = config['normal_size']
  image_to_tensor = transforms.ToTensor()

  # resize image and mask
  images = batch['image']
  new_images = []
  original_sizes = []
  new_sizes = []
  for i in range(len(images)):
    image = images[i]
    original_size = [image.shape[-2], image.shape[-1]]
    original_sizes.append(original_size)
    image, _ = T.resize(image, None, min_size, max_size)
    new_size = [image.shape[-2], image.shape[-1]]
    new_sizes.append(new_size)
    new_images.append(image)
  images = new_images

  # superpoint_targets
  superpoint_targets = {}
  if 'points' in batch:
    points_list = batch['points']
    valid_masks, keypoint_maps, heatmaps = [], [], []
    for i in range(len(images)):
      points = points_list[i].data.cpu().numpy()

      valid_mask = pp.generate_valid_mask(original_sizes[i])
      keypoint_map = pp.generate_keypoint_map(original_sizes[i], points)
      heatmap = pp.generate_heatmap(original_sizes[i], points, gaussian_radius)
      
      valid_mask = torch.tensor(valid_mask)
      keypoint_map = torch.tensor(keypoint_map)
      keypoint_map = keypoint_map.unsqueeze(0)
      heatmap = torch.tensor(heatmap)
      heatmap = heatmap.unsqueeze(0)

      keypoint_maps.append(keypoint_map)
      valid_masks.append(valid_mask)
      heatmaps.append(heatmap)
      superpoint_targets = {'valid_mask':valid_masks, 'kpt_map':keypoint_maps, 'ht':heatmaps}
  superpoint_targets = None if not superpoint_targets else superpoint_targets

  # maskrcnn_targets
  maskrcnn_targets = {}
  keys = ['boxes', 'labels', 'masks']
  for k in keys:
    if k in batch:
      maskrcnn_targets[k] = batch[k]
  maskrcnn_targets = None if not maskrcnn_targets else maskrcnn_targets

  # batch data
  images, new_size = pad_images(images, to_stack=True)
  original_sizes = torch.tensor(original_sizes)
  new_sizes = torch.tensor(new_sizes)
  sizes = {'original_sizes':original_sizes, 'new_sizes':new_sizes}

  return images, sizes, maskrcnn_targets, superpoint_targets


def warp_batch_data(batch, config):
  image_to_tensor = transforms.ToTensor()
  warped_images, warped_masks = [], []
  images = batch['image']
  for i in range(len(images)):
    # warp image and points
    image = tensor_to_numpy(batch['image'][i])
    points = batch['points'][i].data.cpu().numpy()
    warped_image, warped_points, _, H = pp.homographic_augmentation(image, points, config['warped_pair'])
    warped_image = image_to_tensor(warped_image)
    warped_images.append(warped_image)

    # warped masks
    mask = batch['masks'][i]
    warped_mask = warp_batch_images(mask, H)
    warped_masks.append(warped_mask)

  warped_batch = copy.deepcopy(batch)
  warped_batch['image'] = warped_images
  warped_batch['masks'] = warped_masks
  return warped_batch


def extract_points_clusters(points_output, batch_masks):
  '''
  inputs:
    points_output: List[Dict[Tensor]]
    batch_masks: List[Tensor]
  outputs:
    List[List[Tensor]], List[List[Tensor]]
  '''
  assert len(points_output) == len(batch_masks)
  points_clusters, descs_clusters = [], []
  for output, masks in zip(points_output, batch_masks):
    points, descs = output['points'], output['point_descs']
    idx_map = pp.generate_idx_map(points.numpy(), list(masks.shape[-2:]))
    idx_map = torch.tensor(idx_map)
    points_cluster, descs_cluster = [], []
    for m in masks:
      m = m.squeeze()
      cluster_idx_map = m * idx_map
      idxes = cluster_idx_map[cluster_idx_map>0]
      idxes = idxes.long()
      if len(idxes) == 0:
        points_cluster.append(torch.zeros(0, 2))
        descs_cluster.append(torch.zeros(0, 256))
      else:
        points_cluster.append(points[idxes])
        descs_cluster.append(descs[idxes])
    points_clusters.append(points_cluster)
    descs_clusters.append(descs_cluster)

  return points_clusters, descs_clusters

def normalize_points(batch_points, img_sizes):
  '''
  inputs:
    batch_points: List[List[Tensor]
    img_sizes: List[List[H, W]]
  outputs:
    List[List[Tensor]]
  '''  
  new_batch_points = []
  for points_cluster, img_szie in zip(batch_points, img_sizes):
    size = torch.tensor(img_szie).float()
    center = size/2
    norm_points = []
    for points in points_cluster:
      pts = (points - center) / size
      norm_points.append(pts)
    new_batch_points.append(norm_points)
  
  return new_batch_points

def batch_merge(data_list):
  new_data = []
  for data in data_list:
    new_data = new_data + data
  return new_data
      
def select_good_clusters(batch_points, thr=5):
  '''
  inputs:
    batch_points: List[Tensor]
  outputs:
    Tensor[True/False]
  '''
  results = []
  for i in range(len(batch_points)):
    to_keep = True if len(batch_points[i]) >= thr else False
    results.append(to_keep) 
  return torch.tensor(results)

def match_points_clusters(points_output, batch_masks, warped_points_output, warped_batch_masks):
  assert len(batch_masks) == len(warped_batch_masks)
  batch_points, batch_descs = extract_points_clusters(points_output, batch_masks)
  warped_batch_points, warped_batch_descs = extract_points_clusters(warped_points_output, warped_batch_masks)

  original_sizes = [list(img.shape[-2:]) for img in batch_masks]

  batch_points = normalize_points(batch_points, original_sizes)
  warped_batch_points = normalize_points(warped_batch_points, original_sizes)

  batch_points, batch_descs = batch_merge(batch_points), batch_merge(batch_descs)
  warped_batch_points, warped_batch_descs = batch_merge(warped_batch_points), batch_merge(warped_batch_descs)

  keeps = select_good_clusters(batch_points)
  warped_keeps = select_good_clusters(warped_batch_points)
  keeps = keeps * warped_keeps
  
  good_points, good_descs = [], []
  good_warped_points, good_warped_descs = [], []
  for i in range(len(keeps)):
    if keeps[i].item():
      good_points.append(batch_points[i])
      good_descs.append(batch_descs[i])
      good_warped_points.append(warped_batch_points[i])
      good_warped_descs.append(warped_batch_descs[i])
  
  n = len(good_points)
  connections, warped_connections = [], []
  for i in range(n):
    connection = torch.zeros(2*n)
    connection[(i+n)] = 1
    connections.append(connection)
    warped_connection = torch.zeros(2*n)
    warped_connection[i] = 1
    warped_connections.append(warped_connection)

  # merege
  good_points.extend(good_warped_points)
  good_descs.extend(good_warped_descs)
  connections.extend(warped_connections)

  return good_points, good_descs, connections