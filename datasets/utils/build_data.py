#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.coco.coco import COCODataset
from datasets.utils.batch_collator import BatchCollator
from datasets.coco.paths_catalog import DatasetCatalog

def coco_loader(
    data_root, name, config, batch_size=2, transforms=transforms.ToTensor(), drop_last=True,
    remove_images_without_annotations=False, oints_root="", num_workers=8):
  data_info = DatasetCatalog.get(name)

  data_dir = os.path.join(data_root, data_info['args']['root'])
  ann_file = os.path.join(data_root, data_info['args']['ann_file'])

  dataset = COCODataset(
      image_root=data_dir, ann_file=ann_file, config=config, transforms=transforms,
      remove_images_without_annotations=remove_images_without_annotations)
  
  sampler = torch.utils.data.sampler.RandomSampler(dataset)
  batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

  collator = BatchCollator()
  loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator, num_workers=num_workers)

  return loader