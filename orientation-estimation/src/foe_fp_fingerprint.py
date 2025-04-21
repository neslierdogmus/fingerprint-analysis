#!/usr/bin/env python3

from pathlib import Path

from matplotlib import image
import numpy as np

from foe_fp_ground_truth import FOEGroundTruth
from foe_fp_patch import FOEPatch


class FOEFingerprint:
    def __init__(self, base_path, fp_id):
        self.base_path = base_path
        self.fp_id = fp_id

        try:
            img_path = Path(base_path).joinpath(fp_id + '.bmp')
            self.image = image.imread(img_path)
        except FileNotFoundError:
            img_path = Path(base_path).joinpath(fp_id + '.png')
            self.image = image.imread(img_path)

        self.gt = FOEGroundTruth.from_file(base_path, fp_id)
        # TODO: To be updated
        self.fp_type = Path(base_path).parts[-1]

    def __repr__(self):
        return 'FOEFingerprint({}, {})'.format(self.base_path, self.fp_id)

    def __str__(self):
        h, w = self.image.shape
        return '{:3d}x{:3d} {:4s} FOEFingerprint id:{}'.format(w, h,
                                                               self.fp_type,
                                                               self.fp_id)

    def to_patches(self, patch_size=32):
        padded = np.pad(self.image, patch_size//2,
                        mode='constant', constant_values=128)
        h, w = self.gt.orientations.shape
        patches = []
        for r in range(h):
            y = self.gt.border + r * self.gt.step
            for c in range(w):
                x = self.gt.border + c * self.gt.step
                if self.gt.mask[r, c] == 1:
                    patch = FOEPatch(self.fp_id, r, c,
                                     padded[y:y + patch_size,
                                            x:x + patch_size],
                                     self.gt.orientations[r, c],
                                     self.fp_type)
                    patches.append(patch)
        return patches

    def get_patch(self, r, c, patch_size=32):
        padded = np.pad(self.image, patch_size//2,
                        mode='constant', constant_values=128)
        y = self.gt.border + r * self.gt.step
        x = self.gt.border + c * self.gt.step
        patch = FOEPatch(self.filename, r, c,
                         padded[y:y + patch_size, x:x + patch_size],
                         self.gt.orientations[r, c], self.fp_type)
        return patch


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='datasets/Finger/FOESamples/Bad',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-fp', '--fp-id', dest='fp_id',
                        default='00', metavar='FINGERPRINTID',
                        help='id of the fingerprint')
    parser.add_argument('-r', '--radius', dest='radius',
                        default=32, type=int, metavar='R',
                        help='radius for patch extraction')

    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    fp_id = args.fp_id
    radius = args.radius

    fp = FOEFingerprint(base_path, fp_id)
    patches = fp.to_patches(radius)

    print('Created a {} fingerprint with {} patches of size {}x{}'
          .format(fp.fp_type, len(patches), *patches[0].patch.shape))
    print(fp)
