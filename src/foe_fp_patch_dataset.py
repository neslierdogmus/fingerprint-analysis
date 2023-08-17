#!/usr/bin/env python3
import random
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from foe_fingerprint import FOEFingerprint


class FOEFPPatchDataset(Dataset):
    def __init__(self, base_path_list, fp_ids_list, patch_size, n_classes=1):
        super(FOEFPPatchDataset).__init__()
        assert len(base_path_list) == len(fp_ids_list), ("Path and fingerprint"
                                                         " ids lists should "
                                                         " have equal length.")
        self.base_path_list = base_path_list
        self.fp_ids_list = fp_ids_list
        self.n_classes = n_classes
        self.patch_size = patch_size
        self._hflip = False
        self._rotate = False
        self.patches = []
        for i in range(len(base_path_list)):
            base_path = self.base_path_list[i]
            fp_ids = self.fp_ids_list[i]
            for fp_id in fp_ids:
                fp = FOEFingerprint(base_path, fp_id)
                self.patches.extend(fp.to_patches(self.patch_size))

    def __getitem__(self, index):
        foe_patch = self.patches[index]
        x = torch.from_numpy(foe_patch.patch)/255
        x = torch.unsqueeze(x, 0)

        ori = foe_patch.ori

        if self._hflip and random.random() >= 0.5:
            x = tf.hflip(x)
            if ori > 0:
                ori = np.pi - ori

        if self._rotate:
            angle = random.uniform(-20, 20)
            angle_radian = angle / 180 * np.pi
            ori += angle_radian
            if ori >= np.pi:
                ori -= np.pi
            x = tf.rotate(x, angle, interpolation=PIL.Image.BILINEAR)

        x = tf.normalize(x, [0], [1])

        if self.n_classes == 1:
            y = ori
        else:
            # TODO: orientation class is not to be used
            y = ori.class_id(self.n_classes)
            # y = ori.ordinal_code(self.n_classes)

        return x, y, ori, index

    def __len__(self):
        return len(self.patches)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_rotate(self, rotate):
        self._rotate = rotate


if __name__ == '__main__':
    import sys
    import os
    from argparse import ArgumentParser

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    parser = ArgumentParser(description='FOE fingerprint image dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='./datasets/Finger/FOESamples/Bad',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='fingerprint patches of size SxS')
    parser.add_argument('-nc', '--n-classes', dest='n_classes',
                        default=32, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=10, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    patch_size = args.patch_size
    n_classes = args.n_classes
    num_folds = args.num_folds

    index_path = os.path.join(base_path, 'index.txt')
    with open(index_path, 'r') as fin:
        fp_ids = [line.split('.')[0] for line in fin.readlines()[1:]]
    random.shuffle(fp_ids)
    splits = np.array_split(fp_ids, num_folds)

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        foe_ptc_ds_val = FOEFPPatchDataset([base_path], [fp_ids_val],
                                           patch_size, n_classes)
        foe_ptc_ds_tra = FOEFPPatchDataset([base_path], [fp_ids_tra],
                                           patch_size, n_classes)
        foe_ptc_ds_all = FOEFPPatchDataset([base_path, base_path],
                                           [fp_ids_tra, fp_ids_val],
                                           patch_size, n_classes)

        print('Created FOE patch dataset for validation with {} patches'
              .format(len(foe_ptc_ds_val)))
        print(foe_ptc_ds_val.fp_ids_list)

        print('Created FOE image dataset for training with {} patches'
              .format(len(foe_ptc_ds_tra)))
        print(foe_ptc_ds_tra.fp_ids_list)

        print('Created FOE image dataset with {} patches'
              .format(len(foe_ptc_ds_all)))
        print(foe_ptc_ds_all.fp_ids_list)

        print('-' * 80)
