#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np

from foe_fingerprint import FOEFingerprint
from _foe_patch_dataset import FOEPatchDataset


class FOEFingerprintDataset:
    def __init__(self, base_dir, fp_type='good', num_folds=10):
        self.base_dir = base_dir
        self.fp_type = fp_type

        index_path = Path(base_dir).joinpath(fp_type.capitalize(), 'index.txt')
        with open(index_path, 'r') as fin:
            lines = fin.readlines()[1:]
            self.lines = lines
        if fp_type != 'synth':
            # split without shuffling
            self.split_indices = np.array_split(list(range(len(lines))),
                                                num_folds)
        else:
            # all for training
            self.lines = self.lines[:100]
            self.split_indices = [[], [], [], [], []]

    def __repr__(self):
        return 'FOEFingerprintDataset({}, {})'.format(self.base_dir,
                                                      self.fp_type)

    def __str__(self):
        return ('{} FOEFingerprint dataset of size {}'
                .format(self.fp_type.capitalize(), len(self.lines)))

    def __len__(self):
        return len(self.lines)

    def set_split_indices(self, split_dir, split_id, num_folds=10):
        split_dir = Path(split_dir)
        split_name = 'split_{}_{}_{}.npy'.format(split_id, self.fp_type,
                                                 num_folds)
        split_path = split_dir.joinpath(split_name)
        if split_path.exists():
            split_indices = np.load(split_path)
            print('Split indices for {} folds with id {} loaded.'
                  .format(num_folds, split_id))
        else:
            split_dir.mkdir(parents=True, exist_ok=True)
            indices = list(range(len(self.lines)))
            random.shuffle(indices)
            split_indices = np.array_split(indices, num_folds)
            np.save(split_path, split_indices)
            print('Split indices for {} folds created and saved with id {}.'
                  .format(num_folds, split_id))
        self.split_indices = split_indices

    def load_fingerprints(self):
        fingerprints = []
        for line in self.lines:
            items = line.strip().split()
            img_filename = items[0]
            step = int(items[1])
            border = int(items[2])
            fp = FOEFingerprint(self.base_dir, img_filename, self.fp_type)
            assert (fp.gt.border == border)
            assert (fp.gt.step == step)
            fingerprints.append(fp)
        return fingerprints

    def get_patch_datasets(self, fold_ind, radius, patch_size, n_classes=1):
        fp_list = self.load_fingerprints()
        train_patches = []
        val_patches = []
        if self.fp_type != 'synth':
            for idx in range(len(self.split_indices)):
                if idx == fold_ind:
                    split = val_patches
                else:
                    split = train_patches
                for fp_idx in self.split_indices[idx]:
                    split.extend(fp_list[fp_idx].to_patches(radius))
        else:
            for i in range(len(self.lines)):
                train_patches.extend(fp_list[i].to_patches(radius))

        train_dset = FOEPatchDataset(train_patches, patch_size, n_classes)
        val_dset = FOEPatchDataset(val_patches, patch_size, n_classes)

        return train_dset, val_dset


if __name__ == '__main__':
    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-b', '--base-dir', dest='base_dir',
                        default='../datasets/Finger/FOESamples',
                        metavar='BASEDIRECTORY',
                        help='root directory for dataset files')
    parser.add_argument('-fpt', '--fingerprint-type', dest='fp_type',
                        default='bad', metavar='FPTYPE',
                        help='type of the fingerprint images (good or bad)')
    parser.add_argument('-sd', '--split-dir', dest='split_dir',
                        default='../results/splits', metavar='SPLITDIRECTORY',
                        help='root directory for split files')
    parser.add_argument('-si', '--split-id', dest='split_id',
                        default=1, type=int, metavar='SPLITID',
                        help='id of the split to be (created and )used')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=10, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    parser.add_argument('-fi', '--fold-ind', dest='fold_ind',
                        default=1, type=int, metavar='FOLDIND',
                        help='index of the fold to be experimented on')
    parser.add_argument('-r', '--radius', dest='radius',
                        default=32, type=int, metavar='R',
                        help='radius for patch extraction')
    parser.add_argument('-ps', '--patch-size', dest='patch_size',
                        default=32, type=int, metavar='PATCHSIZE',
                        help='patch size to be input to the models')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=8, type=int, metavar='Nc',
                        help='number of classes')

    args = parser.parse_args(sys.argv[1:])

    base_dir = Path(args.base_dir)
    fp_type = args.fp_type
    split_dir = args.split_dir
    split_id = args.split_id
    num_folds = args.num_folds
    fold_ind = args.fold_ind
    radius = args.radius
    patch_size = args.patch_size
    n_classes = args.n_classes

    fpd = FOEFingerprintDataset(base_dir, fp_type)
    print('Created a {} fingerprint dataset with {} fingerprints'
          .format(fpd.fp_type, len(fpd)))

    fpd.set_split_indices(split_dir, split_id, num_folds)
    tset, vset = fpd.get_patch_datasets(fold_ind, radius,
                                        patch_size, n_classes)

    print('Loaded a training set of size {} and a test set of size {}'
          .format(len(tset), len(vset)))
