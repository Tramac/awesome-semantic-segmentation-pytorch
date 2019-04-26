"""Look into Person Dataset"""
import os
import torch
import numpy as np

from PIL import Image
from core.data.dataloader.segbase import SegmentationDataset


class LIPSegmentation(SegmentationDataset):
    """Look into person parsing dataset """

    BASE_DIR = 'LIP'
    NUM_CLASS = 20

    def __init__(self, root='../datasets/LIP', split='train', mode=None, transform=None, **kwargs):
        super(LIPSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _trainval_image_dir = os.path.join(root, 'TrainVal_images')
        _testing_image_dir = os.path.join(root, 'Testing_images')
        _trainval_mask_dir = os.path.join(root, 'TrainVal_parsing_annotations')
        if split == 'train':
            _image_dir = os.path.join(_trainval_image_dir, 'train_images')
            _mask_dir = os.path.join(_trainval_mask_dir, 'train_segmentations')
            _split_f = os.path.join(_trainval_image_dir, 'train_id.txt')
        elif split == 'val':
            _image_dir = os.path.join(_trainval_image_dir, 'val_images')
            _mask_dir = os.path.join(_trainval_mask_dir, 'val_segmentations')
            _split_f = os.path.join(_trainval_image_dir, 'val_id.txt')
        elif split == 'test':
            _image_dir = os.path.join(_testing_image_dir, 'testing_images')
            _split_f = os.path.join(_testing_image_dir, 'test_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + '.png')
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} {} images in the folder {}'.format(len(self.images), split, root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category name."""
        return ('background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
                'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                'rightShoe')


if __name__ == '__main__':
    dataset = LIPSegmentation(base_size=280, crop_size=256)