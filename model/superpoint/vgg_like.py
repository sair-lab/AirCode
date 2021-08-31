from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class VggLike(nn.Module):
  
  def __init__(self, pretrained_net):
    super(VggLike, self).__init__()
    self.pretrained_net = pretrained_net
    self.relu = nn.ReLU(inplace=True)

    c1, c2, h1, h2 = 256, 256, 65, 256
    self.convPa = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.bnPa = nn.BatchNorm2d(c2)
    self.convPb = nn.Conv2d(c2, h1, kernel_size=1, stride=1, padding=0)
    self.bnPb = nn.BatchNorm2d(h1)

    self.convDa = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.bnDa = nn.BatchNorm2d(c2)
    self.convDb = nn.Conv2d(c2, h2, kernel_size=1, stride=1, padding=0)
    self.bnDb = nn.BatchNorm2d(h2)

  def forward(self, x):
  
    output = self.pretrained_net(x)
    x3 = output['x3']

    cPa = self.bnPa(self.relu(self.convPa(x3)))
    semi = self.bnPb(self.convPb(cPa))

    prob = nn.functional.softmax(semi, dim=1)
    prob = prob[:, :-1, :, :]
    prob = nn.functional.pixel_shuffle(prob, 8)

    # descriptor extraction
    cDa = self.bnDa(self.relu(self.convDa(x3)))
    desc_raw = self.bnDb(self.convDb(cDa))
    desc = nn.functional.interpolate(desc_raw, scale_factor=8, mode='bilinear')
    desc = nn.functional.normalize(desc, p=2, dim=1)

    return {'logits': semi, 'prob':prob, 'desc_raw': desc_raw, 'desc': desc}