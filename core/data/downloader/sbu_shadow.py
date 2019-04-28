"""Prepare SBU Shadow datasets"""
import os
import sys
import argparse
import zipfile

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from core.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/sbu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize SBU Shadow dataset.',
        epilog='Example: python sbu_shadow.py --download-dir ~/SBU-shadow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args


#####################################################################################
# Download and extract SBU shadow datasets into ``path``

def download_sbu(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip'),
    ]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite)
        # extract
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(path=path)
        print("Extracted", filename)


if __name__ == '__main__':
    args = parse_args()
    makedirs(os.path.expanduser('~/.torch/datasets'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_sbu(_TARGET_DIR, overwrite=False)
