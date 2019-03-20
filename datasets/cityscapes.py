"""Prepare Cityscapes dataset"""
import os
import errno
import argparse
import zipfile
from utils.download import download

_TARGET_DIR = os.path.expanduser('~/PycharmProjects/Data_zoo/citys')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python prepare_cityscapes.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args


def download_city(path, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ('gtFine_trainvaltest.zip', '99f532cb1af174f5fcc4c5bc8feea8c66246ddbc'),
        ('leftImg8bit_trainvaltest.zip', '2c0b77ce9933cc635adda307fbba5566f5d9d404')]
    download_dir = os.path.join(path, 'downloads')
    try:
        os.makedirs(download_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    for url, checksum in _CITY_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)


if __name__ == '__main__':
    args = parse_args()
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_city(_TARGET_DIR, overwrite=False)
