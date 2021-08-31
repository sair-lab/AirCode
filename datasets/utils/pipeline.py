import os
import cv2
import numpy as np
from random import sample

from datasets.utils import augmentation_legacy as photaug
from datasets.utils import gcn_mask_augmentation as maskaug
from datasets.utils.homographies import sample_homography, warp_points, filter_points

def makedir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
        else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p

def space_to_depth(data, cell_size, add_dustbin=False):
    H, W = data.shape[0], data.shape[1]
    Hc, Wc = H // cell_size, W // cell_size
    result = data[:, np.newaxis, :, np.newaxis]
    result = result.reshape(Hc, cell_size, Wc, cell_size)
    result = np.transpose(result, [1, 3, 0, 2])
    result = result.reshape(1, cell_size ** 2, Hc, Wc)
    result = result.squeeze()
    if add_dustbin:
      dustbin = np.ones([Hc, Wc])
      depth_sum = result.sum(axis=0)
      dustbin[depth_sum>0] = 0
      result = np.concatenate((result, dustbin[np.newaxis, :, :]), axis=0)
    return result

'''
draw gaussian function
'''
def gaussian2D(shape, sigma=1):
  # generate a gaussion in a box
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m+1,-n:n+1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
  return heatmap

def convert_to_guassian(label, radius):
  label_gaussian = np.zeros(label.shape)
  ys, xs = np.where(label > 0)
  if len(xs) != 0:
    for i in range(len(xs)):
      draw_umich_gaussian(label_gaussian, (xs[i], ys[i]), radius)

  return label_gaussian

def generate_shape_gaussian(matrix, radius):
  '''
  Generate 3D or 4D shape like gaussian 
  '''
  origin_shape = matrix.shape
  if len(origin_shape) == 2:
    return convert_to_guassian(matrix, radius)
  elif len(origin_shape) == 3:
    origin_matrix = matrix[np.newaxis, :, :, :]
  elif len(origin_shape) == 4:
    origin_matrix = matrix 
  else:
    return None

  matrix_gaussian = np.zeros(origin_matrix.shape)
  for i in range(matrix_gaussian.shape[0]):
    for j in range(matrix_gaussian.shape[1]):
      matrix_gaussian[i, j, :, :] = convert_to_guassian(origin_matrix[i, j, :, :], radius)
  
  if len(origin_shape) == 3:
    matrix_gaussian = np.squeeze(matrix_gaussian, 0)

  return matrix_gaussian

'''
generate valid mask, heatmap and keypoint map
'''
def generate_valid_mask(img_shape, border_remove=2):
  '''
  inputs :
    img_shape: [H, W]
  '''
  H, W = img_shape[0:2]
  valid_mask = np.zeros((H, W), dtype=np.int)
  valid_mask[border_remove:(H-border_remove), border_remove:(W-border_remove)] = 1
  return valid_mask

def generate_keypoint_map(img_shape, points):
  '''
  inputs :
    img_shape: [H, W]
    points: N * 2, [hy, wx]
  '''
  height, width = img_shape[:2]
  points = (points + 0.5).astype(int)
  points[:, 0] = np.clip(points[:, 0], 0, height - 1)
  points[:, 1] = np.clip(points[:, 1], 0, width -1)
  keypoint_map = np.zeros(img_shape[:2], dtype=np.float32)
  for h, w in points:
    keypoint_map[h, w] = 1.0
  return keypoint_map

def generate_heatmap(img_shape, points, gaussian_radius):
  '''
  inputs:
    img_shape: [H, W]
    points: N * 2, [hy, wx]
    gaussian_radius: int
  '''
  if gaussian_radius < 2:
    heatmap = generate_keypoint_map(img_shape, points)
  else:
    heatmap = np.zeros(img_shape[:2])
    for i in range(points.shape[0]):
      heatmap = draw_umich_gaussian(heatmap, (points[i][1], points[i][0]), gaussian_radius)
  return heatmap

def generate_idx_map(points, shape):
  '''
  inputs:
    image: numpy array, [H, W]
    points: N * 2, [hy, wx]
  '''
  idx_map = np.zeros(shape)
  for i in range(len(points)):
    hy, wx = int(points[i][0]), int(points[i][1])
    idx_map[hy, wx] = i

  return idx_map


'''
homographic augmentation
'''
def homographic_augmentation(image, points, config):
  '''
  inputs:
    image: numpy array
    points: N * 2, [hy, wx]
    config
  '''
  H = sample_homography(image.shape[:2], **config['params'])

  warped_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

  if points.shape[0] > 0:
    warped_points = warp_points(H, points)
    warped_points = filter_points(image.shape[:2], warped_points)
  else:
    warped_points = points

  valid_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
  warped_valid_mask = cv2.warpPerspective(valid_mask, H, (valid_mask.shape[1], valid_mask.shape[0]))
  k = np.ones((config['valid_border_margin'], config['valid_border_margin']), np.uint8)
  warped_valid_mask = cv2.erode(warped_valid_mask, k, iterations=1)
  warped_valid_mask = (warped_valid_mask > 0).astype(int)

  return warped_image, warped_points, warped_valid_mask, H

'''
photometric augmentation
'''
def photometric_augmentation(image, points, config):
  '''
  inputs:
    image: numpy array
    points: N * 2, [hy, wx]
    config
  '''
  primitives = config['primitives']
  fun_name = sample(primitives, 1)[0]
  fun_config = config['params'][fun_name]
  
  if len(image.shape) == 3:
    img = image[:, :, 0]
  else:
    img = image

  aug = getattr(photaug, fun_name)
  img, points = aug(img, np.flip(points, 1), **fun_config)

  img = cv2.merge([img, img, img])
  points = np.flip(points, 1)

  return img, points


'''
mask augmentation
'''
def mask_augmentation(masks, config):
  '''
  inputs:
    image: numpy array or List[numpy array]
    config
  '''
  primitives = config['primitives']

  new_masks = []
  for mask in masks:
    fun_name = sample(primitives, 1)[0]
    fun_config = config['params'][fun_name]
    aug = getattr(maskaug, fun_name)
    mask = aug(mask, **fun_config)
    new_masks.append(mask)

  return np.stack(new_masks)