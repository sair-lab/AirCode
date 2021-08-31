#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')   
from matplotlib import pyplot as plt
import torchvision

def show_batch(batch):
  grid = torchvision.utils.make_grid(batch)
  plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
  plt.title('Batch')
  plt.show()

def show_batch_opencv(batch):
  T = torchvision.transforms.ToTensor()
  batch = [T(img) for img in batch]
  batch = torch.stack(batch)
  show_batch(batch)

def show_numpy(img):
  plt.imshow(img)
  plt.title('Image')
  plt.show()