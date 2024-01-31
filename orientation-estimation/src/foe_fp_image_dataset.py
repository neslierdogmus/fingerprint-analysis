#!/usr/bin/env python3
import random
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from foe_fingerprint import FOEFingerprint


class FOEFPImageDataset(Dataset):
    def __init__(self, base_path_list, fp_ids_list):
        super(FOEFPImageDataset).__init__()
        assert len(base_path_list) == len(fp_ids_list), ("Path and fingerprint"
                                                         " ids lists should "
                                                         " have equal length.")
        self.base_path_list = base_path_list
        self.fp_ids_list = fp_ids_list
        self._hflip = False
        self._rotate = False
        self.fps = []
        for i in range(len(base_path_list)):
            base_path = self.base_path_list[i]
            fp_ids = self.fp_ids_list[i]
            for fp_id in fp_ids:
                fp = FOEFingerprint(base_path, fp_id)
                if fp_id == '02':
                    gt_height = fp.gt.orientations.shape[0]
                    fp.gt.orientations = np.hstack((fp.gt.orientations,
                                                    np.zeros((gt_height, 1))))
                    fp.gt.mask = np.hstack((fp.gt.mask,
                                            np.zeros((gt_height, 1))))
                if fp_id == '07':
                    gt_width = fp.gt.orientations.shape[1]
                    fp.gt.orientations = np.vstack((fp.gt.orientations,
                                                    np.zeros((1, gt_width))))
                    fp.gt.mask = np.vstack((fp.gt.mask,
                                            np.zeros((1, gt_width))))
                self.fps.append(fp)

    def __getitem__(self, index):
        foe_fingerprint = self.fps[index]
        x = foe_fingerprint.image
        mask = foe_fingerprint.gt.mask
        orientations = foe_fingerprint.gt.orientations
        border = foe_fingerprint.gt.border
        step = foe_fingerprint.gt.step

        pad_h = 576-x.shape[0]
        pad_w = 464-x.shape[1]
        x = np.pad(x, ((0, pad_h), (0, pad_w)))
        mask = np.pad(mask, ((0, pad_h//step), (0, pad_w//step)))
        orientations = np.pad(orientations, ((0, pad_h//step),
                                             (0, pad_w//step)))

        omit = int(border-step/2)
        x = np.array(x[omit:-omit, omit:-omit])
        x = torch.from_numpy(x.astype(np.single))

        orientations = torch.from_numpy(orientations.astype(np.single))
        mask = torch.from_numpy(mask.astype(np.single))

        x = torch.unsqueeze(x, 0)
        orientations = torch.unsqueeze(orientations, 0)
        mask = torch.unsqueeze(mask, 0)

        mask_resized = tf.resize(mask, x.shape[1:], interpolation=0)
        x = x * mask_resized
        x_mean = x.sum() / mask_resized.sum()
        x = (x - x_mean) * mask_resized
        x_mean = x.sum() / mask_resized.sum()
        x_var = (((x-x_mean)**2 * mask_resized).sum() / (mask_resized.sum()))
        x_std = x_var**0.5
        x = x / x_std

        if self._hflip and random.random() >= 0.5:
            x = tf.hflip(x)
            for r in range(orientations.shape[1]):
                for c in range(orientations.shape[2]):
                    if orientations[0, r, c] > 0:
                        orientations[0, r, c] = np.pi - orientations[0, r, c]
            orientations = tf.hflip(orientations)
            mask = tf.hflip(mask)

        if self._rotate:
            angle = random.uniform(-20, 20)
            angle_radian = angle / 180 * np.pi
            for r in range(orientations.shape[1]):
                for c in range(orientations.shape[2]):
                    if mask[0, r, c]:
                        orientations[0, r, c] += angle_radian
                        if orientations[0, r, c] >= np.pi:
                            orientations[0, r, c] -= np.pi
                        elif orientations[0, r, c] < 0:
                            orientations[0, r, c] += np.pi
                        if orientations[0, r, c] == np.pi:
                            orientations[0, r, c] = 0
            x = tf.rotate(x, angle, interpolation=PIL.Image.BILINEAR)
            orientations = tf.rotate(orientations, angle)
            mask = tf.rotate(mask, angle)

        mask = mask.squeeze()
        orientations = orientations.squeeze()

        return x, orientations, mask, foe_fingerprint.fp_type, index

    def __len__(self):
        return len(self.fps)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_rotate(self, rotate=True):
        self._rotate = rotate


if __name__ == '__main__':
    import sys
    import os
    from argparse import ArgumentParser

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    parser = ArgumentParser(description='FOE fingerprint image dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='./datasets/foe/Bad',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-nc', '--n-classes', dest='n_classes',
                        default=32, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=10, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
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

        foe_img_ds_val = FOEFPImageDataset([base_path], [fp_ids_val],
                                           n_classes)
        foe_img_ds_tra = FOEFPImageDataset([base_path], [fp_ids_tra],
                                           n_classes)
        foe_img_ds_all = FOEFPImageDataset([base_path, base_path],
                                           [fp_ids_tra, fp_ids_val], n_classes)

        print('Created FOE image dataset for validation with {} fingerprints'
              .format(len(foe_img_ds_val)))
        print(foe_img_ds_val.fp_ids_list)

        print('Created FOE image dataset for training with {} fingerprints'
              .format(len(foe_img_ds_tra)))
        print(foe_img_ds_tra.fp_ids_list)

        print('Created FOE image dataset with {} fingerprints'
              .format(len(foe_img_ds_all)))
        print(foe_img_ds_all.fp_ids_list)

        print('-' * 80)
