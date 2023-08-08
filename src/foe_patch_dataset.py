#!/usr/bin/env python3
import random
import PIL

import torch
from datasets import Dataset
import torchvision.transforms.functional as tf


class FOEPatchDataset(Dataset):
    def __init__(self, patches, patch_size, n_classes=1):
        super(FOEPatchDataset).__init__()
        self.patches = patches.copy()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self._hflip = False
        self.rotate = False
        self._encoder = None
        self._decoder = None
        self.images = sorted(set(p.filename for p in patches))
        self.is_image_good = {image: False for image in self.images}
        for p in self.patches:
            if p.fp_type == 'good':
                self.is_image_good[p.filename] = True

    def __getitem__(self, index):
        foe_patch = self.patches[index]
        x = torch.from_numpy(foe_patch.patch)/255
        x = torch.unsqueeze(x, 0)

        ori = foe_patch.ori

        if self._hflip and random.random() >= 0.5:
            x = tf.hflip(x)
            ori = ori.hflipped()

        if self.rotate:
            new_angle = random.uniform(0, 180)
            r = new_angle - ori.degrees()
            x = tf.rotate(x, r, resample=PIL.Image.BILINEAR)
            ori = ori.rotated(r)

        x = tf.center_crop(x, self.patch_size)
        # x = tf.normalize(x, [0], [1])

        if self._encoder is not None:
            self._encoder.eval()
            with torch.no_grad():
                x = torch.unsqueeze(x, 0)
                x = self._encoder(x)
                if self._decoder:
                    x = self._decoder(x)
                else:
                    x = x.squeeze(0)

        gt_in_radians = ori.radians()
        if self.n_classes == 1:
            y = gt_in_radians.float()
        else:
            # y = ori.class_id(self.n_classes)
            y = ori.ordinal_code(self.n_classes)

        return {
            "input": x,  # Modify as needed
            "label": y,  # Modify as needed
            "gt_in_radians": gt_in_radians,  # Modify as needed
            "index": index  # Modify as needed
        }

    def __len__(self):
        return len(self.patches)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_encoder(self, encoder):
        self._encoder = encoder

    def set_decoder(self, decoder):
        self._decoder = decoder

    def merge(self, pd):
        new_ds = FOEPatchDataset(self.patches, self.patch_size, self.n_classes)
        new_ds.patches += pd.patches
        new_ds.images += pd.images
        new_ds.is_image_good.update(pd.is_image_good)
        return new_ds


if __name__ == '__main__':
    import sys
    import os
    from argparse import ArgumentParser
    from pathlib import Path

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foe_fingerprint_dataset import FOEFingerprintDataset

    parser = ArgumentParser(description='Fingerprint dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='../datasets/Finger/FOESamples',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-nc', '--n-classes', dest='n_classes',
                        default=32, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('-nf', '--n-folds', dest='n_folds',
                        default=5, type=int, metavar='Nf',
                        help='number of folds')
    parser.add_argument('--no-shuffle', dest='no_shuffle',
                        action='store_true',
                        help='do not shuffle fingerprints before'
                        ' fold computation')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    patch_size = args.patch_size
    n_classes = args.n_classes
    n_folds = args.n_folds

    fpd_gd = FOEFingerprintDataset(base_path, 'good', n_folds)
    print('Created a {} fingerprint dataset with {} fingerprints'
          .format(fpd_gd.fp_type, len(fpd_gd)))

    fpd_bd = FOEFingerprintDataset(base_path, 'bad', n_folds)
    print('Created a {} fingerprint dataset with {} fingerprints'
          .format(fpd_bd.fp_type, len(fpd_bd)))

    for f in range(n_folds):
        tset_gd, vset_gd = fpd_gd.getPatchDatasets(f, patch_size,
                                                   patch_size, n_classes)
        tset_bd, vset_bd = fpd_bd.getPatchDatasets(f, patch_size,
                                                   patch_size, n_classes)

        print('''Fold {}/{}
        Good: {} training and {} validation patches
        Bad:  {} training and {} validation patches'''
              .format(f+1, n_folds, len(tset_gd), len(vset_gd),
                      len(tset_bd), len(vset_bd), ))
        print('-' * 80)
