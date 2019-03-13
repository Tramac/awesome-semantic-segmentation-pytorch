from __future__ import division

import torch
import math
import numbers
import random
import sys
import collections
import warnings
import numpy as np
from PIL import Image

from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
else:
    Sequence = collections.abc.Sequence

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Pad", "RandomCrop",
           "RandomHorizontalFlip", "RandomResizedCrop", "RandomRotation"]


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""

    def __call__(self, pic, label):
        img = F.to_tensor(pic)
        label = torch.LongTensor(np.array(label, dtype=np.int))

        return img, label


class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image."""

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic, label):
        img = F.to_pil_image(pic, self.mode)
        label = F.to_pil_image(label, self.mode)

        return img, label


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, label):
        img = F.normalize(img, self.mean, self.std, self.inplace)

        return img, label


class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value."""

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, label):
        label = F.pad(label, self.padding, 0, self.padding_mode)
        if self.fill == -1:
            img = F.pad(img, self.padding, 0, 'reflection')
        else:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
        return img, label


class RandomCrop(object):
    """Crop the given PIL Image at a random location."""

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop."""
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, w, h

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)

        return i, j, tw, th

    def __call__(self, img, label):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, 0, 'constant')

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - img.size[0], 0), 0, 'constant')
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - img.size[1]), 0, 'constant')

        i, j, w, h = self.get_params(img, self.size)
        img = F.crop(img, i, j, w, h)
        label = F.crop(label, i, j, w, h)

        return img, label


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop."""
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[0] - w)
                j = random.randint(0, img.size[1] - h)
                return i, j, w, h

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[0] - w) // 2
        j = (img.size[1] - w) // 2
        return i, j, w, w

    def __call__(self, img, label):
        i, j, w, h = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, w, h, self.size, self.interpolation)
        label = F.resized_crop(label, i, j, w, h, self.size, Image.NEAREST)

        return img, label


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return F.hfilp(img), F.hfilp(label)
        else:
            return img, label


class RandomRotation(object):
    """Rotate the image by angle."""

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation."""
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, label):
        assert img.size == label.size

        w, h = img.size
        angle = self.get_params(self.degrees)

        img = F.pad(img, (w, h, w, h), padding_mode='reflect')
        img = F.rotate(img, angle, resample_mode='bilinear')
        img = F.crop(img, w, h, w, h)

        label = F.pad(label, (w, h, w, h), 0, padding_mode='constant')
        label = F.rotate(label, angle, resample_mode='nearest')
        label = F.crop(label, w, h, w, h)

        return img, label


class Augmentation(object):
    def __init__(self, rotate_degree=0, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.augment = Compose([
            RandomRotation(rotate_degree),
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean, std)
        ])

    def __call__(self, img, label):
        return self.augment(img, label)
