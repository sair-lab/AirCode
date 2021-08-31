#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DescriptorLoss(nn.Module):
  '''
  loss for object descriptor
  '''
  def __init__(self, config):
    super().__init__()
    self.config = config

  def forward(self, descs, conns):
    '''
    descs: N * D
    conns: N * N
    '''
    similarity = torch.einsum('nd,dm->nm', descs, descs.t())  # N * N

    print(similarity)

    pos_idx = conns
    pos_similarity = similarity * pos_idx

    neg_idx0 = torch.ones_like(conns) - conns
    neg_idx0 = neg_idx0 - torch.eye(len(conns), device=conns.device, dtype=conns.dtype)
    neg_similarity0 = similarity * neg_idx0
    value, index = neg_similarity0.topk(1, largest=True)
    value = value.repeat(1, similarity.shape[1])
    neg_idx1 = (neg_similarity0 == value).float()

    zero = torch.tensor(0.0, dtype=similarity.dtype, device=similarity.device)
    positive_dist = torch.max(zero, self.config['train']['positive_margin'] - similarity)
    negative_dist = torch.max(zero, similarity - self.config['train']['negative_margin'])

    ploss = torch.sum(pos_idx * positive_dist) / torch.sum(pos_idx)
    nloss = torch.sum(neg_idx1 * negative_dist) / torch.sum(neg_idx1)

    return ploss, nloss