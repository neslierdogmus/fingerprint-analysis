#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from .foe_mask import FOEMask
from .foe_ground_truth import FOEGroundTruth
from .foe_patch import FOEPatch


class FOEFingerprint:
    def __init__(self, base_dir, filename, is_good=True):
        self.base_dir = base_dir
        self.filename = filename

        img_path = Path(base_dir).joinpath(filename)
        self.image = plt.imread(img_path)

        mask_filename = img_path.stem + '.fg'
        self.mask = FOEMask.from_file(base_dir, mask_filename)

        gt_filename = img_path.stem + '.gt'
        self.gt = FOEGroundTruth.from_file(base_dir, gt_filename)

        self.is_good = is_good

    def __repr__(self):
        return 'FOEFingerprint({}, {}, {})'.format(self.base_dir,
                                                   self.filename,
                                                   self.is_good)

    def __str__(self):
        h, w = self.image.shape
        good_str = 'GOOD' if self.is_good else 'BAD'
        return '{:3d}x{:3d} {:4s} FOEFingerprint from {}'.format(w, h,
                                                                 good_str,
                                                                 self.filename)

    def to_patches(self, radius=32):
        hi, wi = self.image.shape
        padded = np.full((hi + 2*radius, wi + 2*radius),
                         128, dtype=np.uint8)
        padded[radius:radius+hi,
               radius:radius+wi] = self.image
        h, w = self.gt.ori.shape
        patches = []
        for r in range(h):
            y = self.gt.border + r * self.gt.step
            for c in range(w):
                x = self.gt.border + c * self.gt.step
                if self.mask.data[r, c] == 1:
                    patch = FOEPatch(self.filename, r, c,
                                     padded[y:y + 2*radius,
                                            x:x + 2*radius],
                                     self.gt.ori[r, c],
                                     self.is_good)
                    patches.append(patch)
        return patches

    @staticmethod
    def load_index_file(base_dir, filename, is_good):
        index_path = Path(base_dir).joinpath(filename)
        with open(index_path, 'r') as fin:
            lines = fin.readlines()
        fingerprints = []
        for line in lines[1:]:
            items = line.strip().split()
            img_filename = items[0]
            step = int(items[1])
            border = int(items[2])
            fp = FOEFingerprint(base_dir, img_filename, is_good)
            assert(fp.gt.border == border)
            assert(fp.gt.step == step)
            fingerprints.append(fp)
        return fingerprints


if __name__ == '__main__':
    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='/opt/data/FOESamples', metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-r', '--radius', dest='radius',
                        default=32, type=int, metavar='R',
                        help='radius for patch extraction')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    radius = args.radius

    dset_good = FOEFingerprint.load_index_file(base_path.joinpath('Good'),
                                               'index.txt', True)
    dset_bad = FOEFingerprint.load_index_file(base_path.joinpath('Bad'),
                                              'index.txt', False)
    print('Loaded {} Good and {} Bad fingerprints:'.format(len(dset_good),
                                                           len(dset_bad)))
    patches_good = []
    for fp in dset_good:
        print(fp)
        patches_good.extend(fp.to_patches(radius))

    patches_bad = []
    for fp in dset_bad:
        print(fp)
        patches_bad.extend(fp.to_patches(radius))

    print('Created {} Good and {} Bad patches'
          ' of size {}x{}.'.format(len(patches_good),
                                   len(patches_bad),
                                   *patches_good[0].patch.shape))
