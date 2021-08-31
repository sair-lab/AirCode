#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

class GraphAtten(nn.Module):
  def __init__(self, nfeat, nhid, nout, alpha=0.2, nheads=8):
    super(GraphAtten, self).__init__()
    self.attns = [Attention(nfeat, nhid, alpha) for _ in range(nheads)]
    for i, attention in enumerate(self.attns):
        self.add_module('attention_{}'.format(i), attention)

    self.relu = nn.ReLU()

    self.merge = nn.Linear(nheads*nhid, nhid)

    self.mlp1 = nn.Linear((nfeat+nhid), (nfeat+nhid))
    self.bn1 = nn.BatchNorm1d((nfeat+nhid))
    self.mlp2 = nn.Linear((nfeat+nhid), nout)
    self.bn2 = nn.BatchNorm1d(nout)

  def print_para(self, layer):
    model_dict = self.state_dict()
    para = model_dict[layer]
    print("layer = {}".format(para))

  def forward(self, x):
    m = torch.cat([attn(x) for attn in self.attns], dim=1)
    m = self.relu(self.merge(m))
    x = torch.cat([x, m], 1)
    x = self.relu(self.bn1(self.mlp1(x)))
    x = self.relu(self.bn2(self.mlp2(x)))
    return x

    
class Attention(nn.Module):
  def __init__(self, in_features, out_features, alpha):
    super(Attention, self).__init__()
    self.tranq = nn.Linear(in_features, out_features)
    self.trank = nn.Linear(in_features, out_features)
    self.tranv = nn.Linear(in_features, out_features)
    self.norm = nn.Sequential(nn.Softmax(dim=1))
    self.leakyrelu = nn.LeakyReLU(alpha)
    self.relu = nn.ReLU()

  def forward(self, x):
    q = self.relu(self.tranq(x))   # n * dim
    k = self.relu(self.trank(x))   # n * dim
    v = self.relu(self.tranv(x))

    adj = torch.einsum('nd,dm->nm', q, k.t())  # n * n
    adj = self.leakyrelu(adj)
    adj = self.norm(adj)

    m = adj @ v
    return m