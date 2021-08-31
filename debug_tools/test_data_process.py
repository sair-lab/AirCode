#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')   
import os
import yaml
import argparse
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from datasets.utils.preprocess import preprocess_train_data
from datasets.utils import postprocess as post
from datasets.utils.batch_collator import BatchCollator
from debug_tools.show_batch import show_batch, show_numpy, show_batch_opencv
from utils.tools import tensor_to_numpy
from datasets.utils.build_data import coco_loader
from torch.nn import functional as F

from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import paste_masks_in_image

def postT(result,               # type: List[Dict[str, Tensor]]
                image_shapes,         # type: List[Tuple[int, int]]
                original_image_sizes  # type: List[Tuple[int, int]]
                ):
  for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
    boxes = pred["boxes"]
    boxes = resize_boxes(boxes, im_s, o_im_s)
    result[i]["boxes"] = boxes
    if "masks" in pred:
      masks = pred["masks"].unsqueeze(1)
      scale = min(float(o_im_s[0])/im_s[0], float(o_im_s[1])/im_s[1])
      masks = F.interpolate(masks.float(), scale_factor=scale).squeeze(1).byte()
      result[i]["masks"] = masks
  return result


def test(configs):
  # read configs
  model_dir = configs['model_dir']
  data_root = configs['data_root']
  data_config = configs['data']
  train_data_name = data_config['TRAIN']

  debug_dir = "/home/haoyuefan/xk_data/superpoint/coco/debug_results/data_processing"

  # data
  loader = coco_loader(
      data_root=data_root, name=train_data_name, config=data_config, batch_size=2, remove_images_without_annotations=True)

  for iter, batch in enumerate(loader):
    print("iter = {}".format(iter))
    gt = copy.deepcopy(batch)
    original_images = batch['image']
    image_names = batch['image_name']

    images, sizes, maskrcnn_targets, warped_images, superpoint_targets = preprocess_train_data(batch, False, 1, data_config)

    # original_images
    original_images = [tensor_to_numpy(img) for img in original_images]

    # sizes
    original_sizes = sizes['original_sizes']
    new_sizes = sizes['new_sizes']

    # maskrcnn 
    num_images = len(images)
    new_targets = []
    for i in range(num_images):
      target = {}
      num_objs = int(torch.sum(maskrcnn_targets['labels'][i] >= 0).item())
      for k in maskrcnn_targets.keys():
        target[k] = maskrcnn_targets[k][i][:num_objs]
      target['scores'] = torch.ones(num_objs)
      target['masks'] = target['masks'].float()
      new_targets += [target]
    maskrcnn_targets = new_targets
    maskrcnn_targets = postT(maskrcnn_targets, new_sizes.numpy().tolist(), original_sizes.numpy().tolist())

    # superpoint
    points_probs = superpoint_targets['kpt_map']
    points_desc = torch.ones(len(points_probs), 256, points_probs.shape[-2], points_probs.shape[-1])
    points_output = {'prob':points_probs, 'desc': points_desc}

    detections, points_output = post.postprocess(new_sizes, original_sizes, 0.3, maskrcnn_targets, points_output)
    
    results = post.save_detection_results(original_images, image_names, debug_dir, detections, None, points_output, True, True)

    # save gt 
    new_gts = []
    for i in range(len(images)):
      new_gt = {}
      for k in gt:
        new_gt[k] = gt[k][i]
      new_gt['scores'] = new_gt['labels']
      new_gts.append(new_gt)
    
    save_dir_list = [os.path.join(debug_dir, image_name) for image_name in image_names]
    images = copy.deepcopy(original_images)
    images = post.overlay_objects(images, new_gts, None)
    images = post.overlay_points(images, new_gts)
    post.save_images(images, save_dir_list, "groundtruth")


def main():
  parser = argparse.ArgumentParser(description="Test Process")
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
  configs['model_dir'] = args.save_dir
  configs['data_root'] = args.data_root

  test(configs)

if __name__ == "__main__":
  main()