#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from foe_mask import FOEMask
from foe_ground_truth import FOEGroundTruth
from foe_patch import FOEPatch


class FOEFingerprint:
    def __init__(self, base_dir, filename, fp_type):
        self.base_dir = base_dir
        self.filename = filename
        self.subfolder = fp_type.capitalize()

        files_dir = Path(base_dir).joinpath(self.subfolder)

        img_path = files_dir.joinpath(filename)
        self.image = plt.imread(img_path)

        mask_filename = img_path.stem + '.fg'
        self.mask = FOEMask.from_file(files_dir, mask_filename)

        gt_filename = img_path.stem + '.gt'
        self.gt = FOEGroundTruth.from_file(files_dir, gt_filename, fp_type)

        self.fp_type = ['bad', 'good', 'synth'].index(fp_type)

    def __repr__(self):
        return 'FOEFingerprint({}, {}, {})'.format(self.base_dir,
                                                   self.filename,
                                                   self.fp_type)

    def __str__(self):
        h, w = self.image.shape
        type_str = ['bad', 'good', 'synth'][self.fp_type].capitalize()
        return '{:3d}x{:3d} {:4s} FOEFingerprint from {}'.format(w, h,
                                                                 type_str,
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
                                     self.fp_type)
                    patches.append(patch)
        return patches

    def get_patch(self, r, c, radius=32):
        hi, wi = self.image.shape
        padded = np.full((hi + 2*radius, wi + 2*radius), 128, dtype=np.uint8)
        padded[radius:radius+hi, radius:radius+wi] = self.image
        y = self.gt.border + r * self.gt.step
        x = self.gt.border + c * self.gt.step
        patch = FOEPatch(self.filename, r, c,
                         padded[y:y + 2*radius, x:x + 2*radius],
                         self.gt.ori[r, c], self.fp_type)
        return patch


if __name__ == '__main__':
    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='../datasets/Finger/FOESamples/Bad',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-f', '--file-name', dest='filename',
                        default='00.bmp', metavar='FILENAME',
                        help='image file name for the fingerprint')
    parser.add_argument('--fp-type', dest='good',
                        action='store_true',
                        help='fingerprint image is good')
    parser.add_argument('-r', '--radius', dest='radius',
                        default=32, type=int, metavar='R',
                        help='radius for patch extraction')

    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    filename = args.filename
    fp_type = args.fp_type
    radius = args.radius

    fp = FOEFingerprint(base_path, filename, fp_type)
    patches = fp.to_patches(radius)

    print('Created a {} fingerprint with {} patches of size {}x{}'
          .format(fp_type, len(patches), *patches[0].patch.shape))
    print(fp)
