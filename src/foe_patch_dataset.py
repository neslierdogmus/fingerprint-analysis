#!/usr/bin/env python3
import sys
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import PIL

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as TF

from .foe_fingerprint import FOEFingerprint


class FOEPatchDataset(Dataset):
    def __init__(self, patches, n_classes, patch_size):
        super(FOEPatchDataset).__init__()
        self.patches = patches
        self.n_classes = n_classes
        self.patch_size = patch_size
        self._hflip = False
        self._delta_r = None
        self.images = sorted(set(p.filename for p in patches))
        self.is_image_good = {image: False for image in self.images}
        for p in self.patches:
            if p.is_good:
                self.is_image_good[p.filename] = True
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.CenterCrop(patch_size)])
        self.transforms = tfm

    def __getitem__(self, index):
        foe_patch = self.patches[index]
        x = foe_patch.patch
        x = self.transforms(x)

        ori = foe_patch.ori

        if self._hflip and random.random() >= 0.5:
            x = TF.hflip(x)
            ori = ori.hflipped()

        if self._delta_r is not None:
            r = random.uniform(-self._delta_r, self._delta_r)
            x = TF.rotate(x, r,
                          resample=PIL.Image.BILINEAR)
            ori = ori.rotated(r)

        # target = ori.class_id(self.n_classes)
        gt_in_radians = ori.radians()

        return x, torch.FloatTensor([np.sin(gt_in_radians), np.cos(gt_in_radians)]), gt_in_radians

    def __len__(self):
        return len(self.patches)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_delta_r(self, theta_in_radians):
        self._delta_r = theta_in_radians

    @classmethod
    def trainval_for_cv(cls, fp_list, n_classes, patch_size, n_folds, fold):
        RADIUS = patch_size

        # split based on fingerprints not patches.  We run numpy array split on
        # indices rather than the actual list since otherwise the created numpy
        # object array has random ordering implicitly shuffling fingerprints.
        n_fingerprints = len(fp_list)
        fp_splits = np.array_split(range(n_fingerprints), n_folds)
        train_patches = []
        val_patches = []
        for idx in range(n_folds):
            if idx == fold:
                split = val_patches
            else:
                split = train_patches
            for fp_idx in fp_splits[idx]:
                split.extend(fp_list[fp_idx].to_patches(RADIUS))

        train_dset = FOEPatchDataset(train_patches, n_classes, patch_size)
        val_dset = FOEPatchDataset(val_patches, n_classes, patch_size)
        return train_dset, val_dset

    @classmethod
    def trainval_from_split(cls, fp_list, n_classes,
                            patch_size, train_images):
        RADIUS = patch_size

        train_patches = []
        val_patches = []
        for fp in fp_list:
            if fp.filename in train_images:
                train_patches.extend(fp.to_patches(RADIUS))
            else:
                val_patches.extend(fp.to_patches(RADIUS))
        train_dset = FOEPatchDataset(train_patches, n_classes, patch_size)
        val_dset = FOEPatchDataset(val_patches, n_classes, patch_size)
        return train_dset, val_dset
    
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Fingerprint dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='/opt/data/FOESamples', metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-n', '--n-folds', dest='n_folds',
                        default=5, type=int, metavar='Nf',
                        help='number of folds')
    parser.add_argument('--no-shuffle', dest='no_shuffle',
                        action='store_true',
                        help='do not shuffle fingerprints before'
                        ' fold computation')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    patch_size = args.patch_size
    n_folds = args.n_folds

    fp_list = FOEFingerprint.load_index_file(base_path.joinpath('Good'),
                                             'index.txt', True)
    fp_list.extend(FOEFingerprint.load_index_file(base_path.joinpath('Bad'),
                                                  'index.txt', False))
    print('Loaded {} fingerprints.'.format(len(fp_list)))

    if not args.no_shuffle:
        random.shuffle(fp_list)
        print('Randomized splits.')

    for f in range(n_folds):
        tset, vset = FOEPatchDataset.trainval_for_cv(fp_list, 8,
                                                     patch_size,
                                                    n_folds, f)
        print("""Fold {}/{}:
   {} training examples from images:
        {}
   {} validation examples from images:
        {}""".format(f+1, n_folds,
                     len(tset), tset.images,
                     len(vset), vset.images))
        print('-' * 80)
