import collections
import numpy as np
import importlib
import cv2

def get_module(path, name):
  if path == '':
      mod = importlib.import_module(name)
  else:
      mod = importlib.import_module('{}.{}'.format(path, name))
  return getattr(mod, name)

def tensor_to_numpy(image):
  img = image.data.cpu().numpy()
  img = img.transpose(1, 2, 0)
  img = (img * 255.0 + 0.5).astype(np.uint8)
  img = np.clip(img, 0, 255)
  if img.shape[2] == 1:
    img = cv2.merge([img, img, img])
  else:
    img = img.copy()
  return img


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d