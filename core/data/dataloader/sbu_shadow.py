"""SBU Shadow  Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class SBUSegmentation(SegmentationDataset):
    """SBU Shadow Segmentation Dataset
    """
    NUM_CLASS = 2

    def __init__(self, root='../datasets/sbu', split='train', mode=None, transform=None, **kwargs):
        super(SBUSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root)
        self.images, self.masks = _get_sbu_pairs(self.root, self.split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
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

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target > 0] = 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_sbu_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            print(root)
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root, filename)
                    maskname = filename.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'SBUTrain4KRecoveredSmall/ShadowImages')
        mask_folder = os.path.join(folder, 'SBUTrain4KRecoveredSmall/ShadowMasks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split in ('val', 'test')
        img_folder = os.path.join(folder, 'SBU-Test/ShadowImages')
        mask_folder = os.path.join(folder, 'SBU-Test/ShadowMasks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = SBUSegmentation(base_size=280, crop_size=256)