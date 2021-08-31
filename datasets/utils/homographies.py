"""Sample homography matrices
# mimic the function from tensorflow
# very tricky. Need to be careful for using the parameters.

"""
from __future__ import print_function
import sys
sys.path.append('.')
import os
from math import pi
import cv2
import numpy as np
from numpy.random import normal, uniform
from scipy.stats import truncnorm
import torch
from torchvision import transforms
from torchvision.transforms.functional import perspective
import kornia

from utils.tools import dict_update, tensor_to_numpy

homography_adaptation_default_config = {
  'num': 1,
  'aggregation': 'sum',
  'valid_border_margin': 3,
  'homographies': {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.1,
    'perspective_amplitude_x': 0.1,
    'perspective_amplitude_y': 0.1,
    'patch_ratio': 0.5,
    'max_angle': pi,
  },
  'filter_counts': 0
}

def aggregate_prob(prob):
  agg_kernal = torch.ones(1, 3, 3)
  agg_prob = kornia.filters.filter2D(prob, agg_kernal)
  mask = kornia.feature.nms2d(prob, (7, 7), True)

  return agg_prob * mask.to(agg_prob.dtype)
  

def warp_batch_images(images, H):
  '''
  perspective batch images

  input :
    images : pytorch tensors, [..., H, W]
    H : homography matrix, numpy array, [3, 3]

  output :
    pytorch tensors, [..., H, W]
  '''
  def convert_to_list(points):
    points = np.flip(points, 1)
    return points.tolist()

  height, width = images.shape[-2:]
  startpoints = np.array([[0.0, 0.0], [0.0, width - 1.0], [height - 1.0, width - 1.0], [height - 1.0, 0.0]])
  endpoints = warp_points(H, startpoints)
  startpoints = startpoints.astype(int)
  endpoints = (endpoints+0.5).astype(int)
  startpoints = convert_to_list(startpoints)
  endpoints = convert_to_list(endpoints)

  images = perspective(images, startpoints, endpoints)
  return images

def homography_adaptation(image, net, config):
  """Perfoms homography adaptation.
  Inference using multiple random warped patches of the same input image for robust
  predictions.
  Arguments:
      image: A `Tensor` with shape `[N, H, W, 1]`.
      net: A function that takes an image as input, performs inference, and outputs the
          prediction dictionary.
      config: A configuration dictionary containing optional entries such as the number
          of sampled homographies `'num'`, the aggregation method `'aggregation'`.
  Returns:
      A dictionary which contains the aggregated detection probabilities.
  """
  data_type, device = image.device, image.dtype

  prob = net(image)['prob']
  count = torch.ones_like(prob)

  probs = [prob]
  counts = [count]

  config = dict_update(homography_adaptation_default_config, config)
  img_shape = image.shape[-2:]
  transform = transforms.ToTensor()

  for i in range(config['num']):
    H = sample_homography(img_shape, **config['homographies'])
    H_inv = np.linalg.inv(H)

    warped_img = warp_batch_images(image, H)
    warped_prob = net(warped_img)['prob']

    valid_mask = warp_batch_images(torch.ones_like(warped_prob), H)
    count = warp_batch_images(torch.ones_like(warped_prob), H_inv)
    warped_prob = warped_prob * valid_mask

    probs_proj = warp_batch_images(warped_prob, H_inv)
    probs_proj = probs_proj * count

    probs += [probs_proj]
    counts += [count]

  probs = torch.sum(torch.stack(probs, 0), 0)
  counts = torch.sum(torch.stack(counts, 0), 0)

  probs = probs/counts
  probs = aggregate_prob(probs)

  return {'prob': probs} 

def sample_homography(
    shape, perspective=True, scaling=True, rotation=True, translation=True,
    n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
    allow_artifacts=False, translation_overflow=0.):
  """Sample a random valid homography.

  Computes the homography transformation between a random patch in the original image
  and a warped projection with the same image size.
  The original patch, which is initialized with a simple half-size centered crop, is
  iteratively projected, scaled, rotated and translated.

  Arguments:
    shape: A numpy array specifying the height and width of the original image.
    perspective: A boolean that enables the perspective and affine transformations.
    scaling: A boolean that enables the random scaling of the patch.
    rotation: A boolean that enables the random rotation of the patch.
    translation: A boolean that enables the random translation of the patch.
    n_scales: The number of tentative scales that are sampled when scaling.
    n_angles: The number of tentatives angles that are sampled when rotating.
    scaling_amplitude: Controls the amount of scale.
    perspective_amplitude_x: Controls the perspective effect in x direction.
    perspective_amplitude_y: Controls the perspective effect in y direction.
    patch_ratio: Controls the size of the patches used to create the homography.
    max_angle: Maximum angle used in rotations.
    allow_artifacts: A boolean that enables artifacts when applying the homography.
    translation_overflow: Amount of border artifacts caused by translation.

  Returns:
    opencv homography transform.
  """

  # Corners of the output image
  margin = (1 - patch_ratio) / 2
  pts1 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])
          
  # Corners of the input patch
  pts2 = pts1.copy()

  # Random perspective and affine perturbations
  # lower, upper = 0, 2
  std_trunc = 2

  if perspective:
    if not allow_artifacts:
      perspective_amplitude_x = min(perspective_amplitude_x, margin)
      perspective_amplitude_y = min(perspective_amplitude_y, margin)
    perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
    h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
    h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
    pts2 += np.array([[h_displacement_left, perspective_displacement],
                      [h_displacement_left, -perspective_displacement],
                      [h_displacement_right, perspective_displacement],
                      [h_displacement_right, -perspective_displacement]]).squeeze()

  # Random scaling
  # sample several scales, check collision with borders, randomly pick a valid one
  if scaling:
    scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
    scales = np.concatenate((np.array([1]), scales), axis=0)
    center = np.mean(pts2, axis=0, keepdims=True)
    scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
    if allow_artifacts:
      valid = np.arange(n_scales)  # all scales are valid except scale=1
    else:
      valid = (scaled >= 0.) * (scaled < 1.)
      valid = valid.prod(axis=1).prod(axis=1)
      valid = np.where(valid)[0]
    idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
    pts2 = scaled[idx,:,:]

  # Random translation
  if translation:
    t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
    if allow_artifacts:
      t_min += translation_overflow
      t_max += translation_overflow
    pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

  # Random rotation
  # sample several rotations, check collision with borders, randomly pick a valid one
  if rotation:
    angles = np.linspace(-max_angle, max_angle, num=n_angles)
    angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
    center = np.mean(pts2, axis=0, keepdims=True)
    rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                    np.cos(angles)], axis=1), [-1, 2, 2])
    rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
    if allow_artifacts:
      valid = np.arange(n_angles)  # all scales are valid except scale=1
    else:
      valid = (rotated >= 0.) * (rotated < 1.)
      valid = valid.prod(axis=1).prod(axis=1)
      valid = np.where(valid)[0]
    idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
    pts2 = rotated[idx,:,:]

  # Rescale to actual size
  shape = shape[::-1]  # different convention [y, x]
  pts1 *= shape
  pts2 *= shape

  homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
  return homography

def warp_points(H, points):
  '''
  perspective points

  input :
    H : homography matrix
    points : numpy array. N * 2, (hy, wx)

  output :
    numpy array, N * 2, (hy, wx)
  '''
  points = np.flip(points, 1)
  points = points[np.newaxis, :, :]
  points = cv2.perspectiveTransform(points, H)
  points = np.squeeze(points, 0)
  return np.flip(points, 1)

def warp_points_bacth(H_list, points):
  '''
  perspective points to bacth points

  input :
    H_list : homography matrix list, numpy array, B * 3 * 3
    points : numpy array. N * 2, (hy, wx)

  output :
    numpy array, B * N * 2, (b, hy, wx)
  '''
  assert len(H_list.shape) == 3

  batch_points =  np.zeros([0, points.shape[0], points.shape[1]], dtype = points.dtype)
  for i in range(H_list.shape[0]):
    warped_points = warp_points(H_list[i], points)
    batch_points = np.concatenate((batch_points, warped_points[np.newaxis, :, :]), axis = 0)

  return batch_points

def filter_points(img_shape, points, border=4):
  '''
  filter points

  input :
    img_shape : list, [H, W]
    points : numpy array. N * 2, (hy, wx)
    border : border

  output :
    numpy array, M * 2, (ht, wx)
  '''
  H, W = img_shape
  points = (points + 0.5).astype(int)
  toremoveW = np.logical_or(points[:, 1] < border, points[:, 1] >= (W-border))
  toremoveH = np.logical_or(points[:, 0] < border, points[:, 0] >= (H-border))
  toremove = np.logical_or(toremoveW, toremoveH)
  points = points[~toremove, :]
  return points


if __name__ == "__main__":
  from debug_tools.show_batch import show_numpy

  def draw_points(points, img):
    for j in range(points.shape[0]):
      y = points[j][0].astype(int)
      x = points[j][1].astype(int)
      if x < 0:
        break;
      cv2.circle(img, (x,y), 1, (255,0,0), thickness=-1)
    return img

  data_root = "/media/xukuan/0000678400004823/dataset/superpoint/synthetic_dataset/test"
  image_dir = "images"
  point_dir = "points"
  image_name = "67.png"
  point_name = "67.txt"
  image_path = os.path.join(data_root, image_dir, image_name)
  point_path = os.path.join(data_root, point_dir, point_name)

  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  points = np.loadtxt(point_path, dtype=np.float32, ndmin=2)

  if len(image.shape) == 2:
    image = cv2.merge([image, image, image])

  show_numpy(image)

  image_shape = np.array(image.shape[:2])
  H = sample_homography(shape=image_shape, perspective=True, scaling=True, rotation=True, translation=True)
  perspective_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0])) 
  points = warp_points(H, points)
  points = filter_points(image_shape, points)
  image_show = draw_points(points, perspective_img)
  show_numpy(image_show)
