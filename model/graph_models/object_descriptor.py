#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

from model.graph_models.attention import GraphAtten

class ObjectDescriptor(nn.Module):
  def __init__(self, config):
    super(ObjectDescriptor, self).__init__()
    points_encoder_dims = config['points_encoder_dims']
    descriptor_dim = config['descriptor_dim']
    nhid = config['hidden_dim']
    alpha = config['alpha']
    nheads = config['nheads']
    nout = config['nout']
    nfeat = descriptor_dim + points_encoder_dims[-1]
    self.points_encoder = PointsEncoder(points_encoder_dims)
    self.gcn = GCN(nfeat, nhid, nout, alpha, nheads)

  def forward(self, batch_points, batch_descs):
    '''
    inputs:
      batch_points: List[Tensor], normalized points, each tensor belong to a object
      batch_descs: List[Tensor]
    '''
    batch_features, locations = [], []
    for points, descs in zip(batch_points, batch_descs):
      encoded_points = self.points_encoder(points)
      features = torch.cat((descs, encoded_points), dim=1)
      features, w = self.gcn(features)
      batch_features.append(features)
      locations.append(w)
    batch_features = torch.stack(batch_features)
    batch_features = nn.functional.normalize(batch_features, p=2, dim=-1)
    locations = torch.cat(locations, 0)
    return batch_features, locations


class PointsEncoder(nn.Module):
  def __init__(self, dims):
    super(PointsEncoder, self).__init__()  
    layers = []
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i], dims[i+1]))
      if i != len(dims)-2:
        layers.append(nn.BatchNorm1d((dims[i+1])))
        layers.append(nn.ReLU())

    self.layers = layers
    for i, layer in enumerate(self.layers):
      self.add_module('point_encoder{}'.format(i), layer)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x


class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nout, alpha=0.2, nheads=8):
    super(GCN, self).__init__()

    self.atten1 = GraphAtten(nfeat, nhid, nfeat, alpha, nheads)
    self.atten2 = GraphAtten(nfeat, nhid, nfeat, alpha, nheads)
    self.tran1 = nn.Linear(nfeat, nfeat)
    self.relu = nn.ReLU()
    self.sparsification = Sparsification(nfeat, nout)

  def forward(self, x):
    x = self.atten1(x)
    x = self.atten2(x)
    x = self.relu(self.tran1(x))
    x, w = self.sparsification(x)

    return x, w


class Sparsification(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(Sparsification, self).__init__()

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    self.location_encoder1 = nn.Linear(input_dim, input_dim)
    self.location_encoder2 = nn.Linear(input_dim, output_dim)

    self.feature_encoder1 = nn.Linear(input_dim, input_dim)
    self.feature_encoder2 = nn.Linear(input_dim, output_dim)
    self.feature_encoder3 = nn.Linear(output_dim, output_dim)


  def forward(self, x):


    descriptor = self.relu(self.feature_encoder1(x))
    descriptor = self.relu(self.feature_encoder2(descriptor))

    locations = self.relu(self.location_encoder1(x))
    locations = self.relu(self.location_encoder2(locations))
    norm_locations = nn.functional.normalize(locations, p=2, dim=-1)

    descriptor = locations * descriptor
    descriptor = torch.sum(descriptor, 0)
    descriptor = self.feature_encoder3(descriptor)

    return descriptor, norm_locations