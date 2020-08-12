
import cv2
import numpy as np
import torch
import random
import scipy.ndimage as ndimage
from IPython import embed
from config import *


class CropCenter(object):   # object is center is better
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target, mask):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        assert x1 >= 0 and y1 >= 0, "Crop size must larger than size"
        inputs = inputs[y1 : y1 + th, x1 : x1 + tw]
        target = target[y1 : y1 + th, x1 : x1 + tw]
        mask   = mask[y1 : y1 + th, x1 : x1 + tw]
        return inputs, target, mask


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, depth, mask):
        if np.random.randn() > 0.5:
            image = image[:,::-1,:]
            depth = depth[:,::-1]
            mask = mask[:,::-1]

        return image, depth, mask


class Resize(object):
    def __init__(self, size=(256,256)):
        self.size = size
    def __call__(self, image, depth, mask):
        image = cv2.resize(image, self.size,interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, self.size,interpolation=cv2.INTER_NEAREST)
        mask  = cv2.resize(mask,  self.size,interpolation=cv2.INTER_NEAREST)
        return image, depth, mask


class ToTensor(object):
    def __call__(self, image, depth, mask):
        image = torch.from_numpy(image.astype('float32')).permute(2, 0, 1)
        depth =  torch.from_numpy(depth[None,:,:].astype('float32'))
        mask = torch.from_numpy(mask[None,:,:].astype('float32'))
        return image, depth, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, depth, mask):
        for t in self.transforms:
            img, depth, mask = t(img, depth, mask)
        return img, depth, mask


class ConvertFromInts(object):
    def __call__(self, image, depth, mask):
        return image.astype(np.float32), depth, mask


class Normalize(object):
    def __call__(self, image, depth, mask):
        image = image.astype(np.float32)
        image /= 255.0
        return image*4., depth, mask

