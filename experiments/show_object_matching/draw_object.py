#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os

import torch
from torchvision import transforms
import yaml
import cv2
import numpy as np
import argparse
import copy

from utils.tools import tensor_to_numpy
from utils import cv2_util


def compute_colors_for_labels(labels):
  """
  Simple function that adds fixed colors depending on the class
  """
  palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = labels[:, None] * palette
  colors = (colors % 255).numpy().astype("uint8")
  return colors


def overlay_boxes(image, boxes, colors):
  """
  Adds the predicted boxes on top of the image

  Arguments:
      image (np.ndarray): an image as returned by OpenCV
  """

  for box, color in zip(boxes, colors):
    box = box.to(torch.int64)
    top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    image = cv2.rectangle(
        image, tuple(top_left), tuple(bottom_right), tuple(color), 1
    )

  return image


def overlay_class_names(image, boxes, textes, colors):
  """
  Adds detected class names and scores in the positions defined by the
  top-left corner of the predicted bounding box

  Arguments:
      image (np.ndarray): an image as returned by OpenCV
  """

  for box, text, color in zip(boxes, textes, colors):
    x, y = (box[0] + box[2]) / 2 - 100, (box[1] + box[3]) / 2
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2
    )

  return image


def overlay_mask(image, masks, colors):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    for mask, color in zip(masks, colors):
      if len(mask.shape) == 3:
        mask = mask.squeeze(0)
      thresh = tensor_to_numpy(mask[None, :, :])
      thresh = thresh[:, :, 0]
      contours, hierarchy = cv2_util.findContours(
          thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
      )
      image = cv2.drawContours(image, contours, -1, color, 4)

    composite = image

    return composite


def draw_object(data, colors=None, match_idx_list=None):
  image = data['image']
  points = data['points']
  objects = data['objects']
  labels = objects['labels']
  boxes = objects['boxes']
  masks = objects['masks']
  if colors is None:
    colors = compute_colors_for_labels(labels).tolist()
  
  # image = overlay_boxes(image, boxes, colors)
  image = overlay_mask(image, masks, colors)

  textes = []
  for idx in range(len(boxes)):
    if idx in match_idx_list:
      i = match_idx_list.index(idx)
      text = "object" + str(i+1)
    else:
      text = "no matching object"
    textes.append(text)
  image = overlay_class_names(image, boxes, textes, colors)

  H, W = image.shape[:2]

  for p, c in zip(points, colors):
    p = p.cpu().numpy()
    if len(p) == 0:
      continue
    for i in range(len(p)):
      x = round(p[i][1] * W + W/2)
      y = round(p[i][0] * H + H/2)
      if x < 0:
        continue
      cv2.circle(image, (x,y), 7, tuple(c), thickness=-1)

  return image, colors