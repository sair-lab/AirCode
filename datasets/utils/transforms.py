import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from typing import List, Tuple, Dict, Optional

@torch.jit.unused
def _resize_image_and_masks_onnx(image, self_min_size, self_max_size, target):
  # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
  from torch.onnx import operators
  im_shape = operators.shape_as_tensor(image)[-2:]
  min_size = torch.min(im_shape).to(dtype=torch.float32)
  max_size = torch.max(im_shape).to(dtype=torch.float32)
  scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

  image = torch.nn.functional.interpolate(
      image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
      align_corners=False)[0]

  if target is None:
    return image, target

  if "masks" in target:
    mask = target["masks"]
    mask = F.interpolate(mask[:, None].float(), scale_factor=scale_factor)[:, 0].byte()
    target["masks"] = mask
  return image, target


def _resize_image_and_masks(image, self_min_size, self_max_size, target):
  # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
  im_shape = torch.tensor(image.shape[-2:])
  min_size = float(torch.min(im_shape))
  max_size = float(torch.max(im_shape))
  scale_factor = self_min_size / min_size
  if max_size * scale_factor > self_max_size:
    scale_factor = self_max_size / max_size
  image = torch.nn.functional.interpolate(
      image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
      align_corners=False)[0]

  if target is None:
    return image, target

  if "masks" in target:
    mask = target["masks"]
    mask = F.interpolate(mask[:, None].float(), scale_factor=scale_factor)[:, 0].byte()
    target["masks"] = mask
  return image, target

def normalize(image, image_mean, image_std):
  dtype, device = image.dtype, image.device
  mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
  std = torch.as_tensor(image_std, dtype=dtype, device=device)
  return (image - mean[:, None, None]) / std[:, None, None]

def resize(image, target, min_size, max_size):
  # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
  h, w = image.shape[-2:]
  if torchvision._is_tracing():
    image, target = _resize_image_and_masks_onnx(image, min_size, float(max_size), target)
  else:
    image, target = _resize_image_and_masks(image, min_size, float(max_size), target)

  if target is None:
    return image, target

  bbox = target["boxes"]
  bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
  target["boxes"] = bbox

  if "keypoints" in target:
    keypoints = target["keypoints"]
    keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
    target["keypoints"] = keypoints
  return image, target

# _onnx_batch_images() is an implementation of
# batch_images() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_batch_images(images, size_divisible=32):
  # type: (List[Tensor], int) -> Tensor
  max_size = []
  for i in range(images[0].dim()):
    max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
    max_size.append(max_size_i)
  stride = size_divisible
  max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
  max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
  max_size = tuple(max_size)

  # work around for
  # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
  # which is not yet supported in onnx
  padded_imgs = []
  for img in images:
    padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
    padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
    padded_imgs.append(padded_img)

  return torch.stack(padded_imgs)

def resize_keypoints(keypoints, original_size, new_size):
  # type: (Tensor, List[int], List[int]) -> Tensor
  ratios = [
      torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
      torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
      for s, s_orig in zip(new_size, original_size)
  ]
  ratio_h, ratio_w = ratios
  resized_data = keypoints.clone()
  if torch._C._get_tracing_state():
    resized_data_0 = resized_data[:, :, 0] * ratio_w
    resized_data_1 = resized_data[:, :, 1] * ratio_h
    resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
  else:
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
  return resized_data


def resize_boxes(boxes, original_size, new_size):
  # type: (Tensor, List[int], List[int]) -> Tensor
  ratios = [
    torch.tensor(s, dtype=torch.float32, device=boxes.device) /
    torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
    for s, s_orig in zip(new_size, original_size)
  ]
  ratio_height, ratio_width = ratios
  xmin, ymin, xmax, ymax = boxes.unbind(1)

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  return torch.stack((xmin, ymin, xmax, ymax), dim=1)
