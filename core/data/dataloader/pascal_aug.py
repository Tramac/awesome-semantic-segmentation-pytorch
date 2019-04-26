"""Pascal Augmented VOC Semantic Segmentation Dataset."""
import os
import torch
import scipy.io as sio
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class VOCAugSegmentation(SegmentationDataset):
    """Pascal VOC Augmented Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCAugSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOCaug/dataset/'
    NUM_CLASS = 21

    def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
        super(VOCAugSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if split == 'train':
            _split_f = os.path.join(_voc_root, 'trainval.txt')
        elif split == 'val':
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self._load_mat(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, target, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _load_mat(self, filename):
        mat = sio.loadmat(filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')


if __name__ == '__main__':
    dataset = VOCAugSegmentation()