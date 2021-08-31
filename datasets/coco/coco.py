#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os

import torch
import torchvision
import cv2
import numpy as np

from structures.segmentation_mask import SegmentationMask
from datasets.utils import pipeline as pp
from datasets.utils import transforms as T

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
  return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
  return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
  # if it's empty, there is no annotation
  if len(anno) == 0:
    return False
  # if all boxes have close to zero area, there is no annotation
  if _has_only_empty_bbox(anno):
    return False
  # keypoints task have a slight different critera for considering
  # if an annotation is valid
  if "keypoints" not in anno[0]:
    return True
  # for keypoint detection tasks, only consider valid images those
  # containing at least min_keypoints_per_image
  if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    return True
  return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
  def __init__(
      self, image_root, ann_file, config, remove_images_without_annotations, 
      transforms=None
  ):
    super(COCODataset, self).__init__(image_root, ann_file)
    # sort indices for reproducible results
    self.ids = sorted(self.ids)

    # filter images without detection annotations
    if remove_images_without_annotations:
      ids = []
      for img_id in self.ids:
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = self.coco.loadAnns(ann_ids)
        if has_valid_annotation(anno):
          ids.append(img_id)
      self.ids = ids

    self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

    self.json_category_id_to_contiguous_id = {
      v: i + 1 for i, v in enumerate(self.coco.getCatIds())
    }
    self.contiguous_category_id_to_json_id = {
      v: k for k, v in self.json_category_id_to_contiguous_id.items()
    }
    self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
    self._transforms = transforms

    # for superpoint
    self.length = len(self.ids)
    self.config = config
    self.points_root = image_root + "_points"

  def __getitem__(self, idx):
    data = {}

    # image
    image, anno = super(COCODataset, self).__getitem__(idx)
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    image = cv2.merge([image, image, image])
    image = self._transforms(image)
    data['image'] = image

    # for maskrcnn
    # filter crowd annotations
    # TODO might be better to add an extra field
    anno = [obj for obj in anno if obj["iscrowd"] == 0]
    boxes = [obj["bbox"] for obj in anno]
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes, xywh
    # remove small bbox
    keep = (boxes[:, 2] > 4) & (boxes[:, 3] > 4)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
    data['boxes'] = boxes[keep]  # [x1, y1, x2, y2]

    labels = [obj["category_id"] for obj in anno]
    labels = [self.json_category_id_to_contiguous_id[c] for c in labels]
    labels = torch.tensor(labels)
    data['labels'] = labels[keep]

    if anno and "segmentation" in anno[0]:
      masks = [obj["segmentation"] for obj in anno]
      masks = SegmentationMask(masks, (image.shape[2], image.shape[1]), mode='poly')
      masks = masks.get_mask_tensor()
      masks = masks
      if len(masks.shape) == 2:
        masks = masks.unsqueeze(0)
      data['masks'] = masks[keep]

    # for superpoint
    image_info = self.get_img_info(idx)
    image_name = image_info['file_name'].split('.')[0]
    data['image_name'] = image_name

    point_name = image_name + ".txt"
    point_path = os.path.join(self.points_root, point_name)
    points = np.loadtxt(point_path, dtype=np.float32, ndmin=2)
    if np.sum(points) < 0:
      points = np.empty((0, 2), dtype=np.float32)
    points = torch.tensor(points)
    data['points'] = points
  
    return data

  def get_img_info(self, index):
    img_id = self.id_to_img_map[index]
    img_data = self.coco.imgs[img_id]
    return img_data
  
if __name__ == "__main__":
  
  import torchvision.transforms as transforms
  from debug_tools.show_batch import show_batch, show_numpy
  from torch.utils.data import Dataset, DataLoader
  from datasets.utils.batch_collator import BatchCollator
  import yaml

  root = "/home/haoyuefan/xk_data/superpoint/coco/full/coco/train2014"
  annFile = "/home/haoyuefan/xk_data/superpoint/coco/full/coco/annotations/instances_train2014.json"
  config = "/home/xukuan/code/object_rcnn/config/train_superpoint_coco.yaml"

  f = open(config, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)

  dataset = COCODataset(root, annFile, configs['data'], True, transforms=transforms.ToTensor())

  print(dataset.categories)