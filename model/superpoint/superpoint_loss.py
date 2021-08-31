#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F

from datasets.utils.homographies import warp_points_bacth


def detector_loss(pred, heatmap, valid_mask):
  ''' 
  Modified focal loss. Exactly the same as CornerNet.
  Runs faster and costs a little bit more memory
  inputs:
    pred: batch * c * h * w
    heatmap: batch * c * h * w
    valid_mask: batch * c * h * w
  '''
  pos_inds = heatmap.eq(1).float()
  neg_inds = heatmap.lt(1).float()

  neg_weights = torch.pow(1 - heatmap, 4)

  loss = 0
  eps = 1e-7

  num_pos = pos_inds.float().sum()

  pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds * valid_mask
  neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds * valid_mask
 
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  
  return loss


def descriptor_loss(descriptors, warped_descriptors, homographies,
                    valid_mask, warped_valid_mask, **config):
    # Compute the position of the center pixel of every cell in the image
    batch_size, Dc, Hc, Wc = descriptors.shape
    coord_cells = np.stack(np.meshgrid(range(Hc), range(Wc), indexing='ij'), axis=-1)
    coord_cells = coord_cells * config['cell'] + config['cell'] // 2 # (Hc, Wc, 2)
    coord_cells = coord_cells.astype(float)
    # coord_cells is now a grid containing the coordinates of the Hc * Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    H_list = np.squeeze(homographies.cpu().numpy(), axis = 1) 
    warped_coord_cells = warp_points_bacth(H_list, np.reshape(coord_cells, [-1, 2]))
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc * Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = np.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2])  # represent warped_image coord_cells
    warped_coord_cells = np.reshape(warped_coord_cells, [batch_size, Hc, Wc, 1, 1, 2]) # represent oridin image coord_cells

    cell_distances = coord_cells - warped_coord_cells

    cell_distances = np.linalg.norm(cell_distances, axis=-1)

    # s = np.less_equal(cell_distances, config['cell'] - 0.5).astype(float)
    # s = torch.tensor(s, dtype=descriptors.dtype, device=descriptors.device)
    s = (cell_distances <= config['cell'] - 0.5)
    s = torch.tensor(s, device=descriptors.device)
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['cell']
    # and 0 otherwise

    # valid_mask
    normalization = torch.sum(warped_valid_mask).float()
    valid_mask = torch.nn.functional.interpolate(valid_mask.unsqueeze(1).float(), scale_factor=1.0/config['cell'], mode='bilinear')
    warped_valid_mask = torch.nn.functional.interpolate(warped_valid_mask.unsqueeze(1).float(), scale_factor=1.0/config['cell'], mode='bilinear')

    valid_mask = valid_mask.squeeze(1) > 0.5
    warped_valid_mask = warped_valid_mask.squeeze(1) > 0.5

    valid_mask = torch.reshape(valid_mask, [batch_size, Hc, Wc, 1, 1])
    warped_valid_mask = torch.reshape(warped_valid_mask, [batch_size, 1, 1, Hc, Wc])
    valid_mask = valid_mask * warped_valid_mask

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = descriptors.permute(0, 2, 3, 1) # B * C * H * W -> B * H * W *C
    descriptors = torch.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1]) # B * Hc * Wc * 1 * 1 * 256
    descriptors = F.normalize(descriptors, dim=-1)

    warped_descriptors = warped_descriptors.permute(0, 2, 3, 1) # B * C * H * W -> B * H * W *C
    warped_descriptors = torch.reshape(warped_descriptors, [batch_size, 1, 1, Hc, Wc, -1]) # B * 1 * 1 * Hc * Wc * 256
    warped_descriptors = F.normalize(warped_descriptors, dim=-1)

    dot_product_desc = (warped_descriptors * descriptors).sum(dim=-1) # B * Hc * Wc * Hc * Wc
    dot_product_desc = F.relu(dot_product_desc) # B * Hc * Wc * Hc * Wc

    zero = torch.tensor(0.0, dtype=descriptors.dtype, device=descriptors.device)

    positive_dist = torch.max(zero, config['train']['positive_margin'] - dot_product_desc)
    negative_dist = torch.max(zero, dot_product_desc - config['train']['negative_margin'])

    loss = (config['train']['lambda_d'] * s * positive_dist + (~s) * negative_dist) * valid_mask
    loss = torch.sum(loss)/normalization

    return loss


class SuperPointLoss(torch.nn.Module):
  '''
  loss for magicpoint: detector loss
  '''
  def __init__(self, config):
    super(SuperPointLoss, self).__init__()
    self.detector_loss = detector_loss
    self.descriptor_loss = descriptor_loss
    self.config = config

  def forward(self, inputs, outputs):
    loss = self.detector_loss(outputs['outputs']['prob'], inputs['ht'], inputs['valid_mask'])
    loss_dict = {'points_loss': loss}
    if 'warped_image' in inputs:
      warped_loss = self.detector_loss(outputs['warped_outputs']['prob'], inputs['warped_ht'], inputs['warped_valid_mask'])

      loss_dict['warped_points_loss'] = warped_loss
      loss = loss + warped_loss
      if self.config['train']['add_descriptor']:
        descriptor_loss = self.descriptor_loss(outputs['outputs']['desc_raw'], outputs['warped_outputs']['desc_raw'], 
            inputs['H'], inputs['valid_mask'], inputs['warped_valid_mask'], **self.config)
        loss_dict['descriptor_loss'] = descriptor_loss
        loss = loss + self.config['train']['lambda_loss'] * descriptor_loss

    return loss, loss_dict
