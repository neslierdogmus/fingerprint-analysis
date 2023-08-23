#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from fc_fingerprint import FCFingerprint


class FCFPDataset(Dataset):
    def __init__(self, base_path, fp_ids, n_classes=5):
        super(FCFPDataset).__init__()

        self.base_path = base_path
        self.fp_ids = fp_ids
        self.n_classes = n_classes
        self._hflip = False
        self._rotate = False
        self._resize = False
        self.fps = []

        for fp_id in fp_ids:
            fp_folder = 'figs_'+str((fp_id - 1) // 250)
            fp_path = Path(base_path).joinpath('png_txt', fp_folder)
            fp_id = str(fp_id).zfill(4)
            for fp_img_path in fp_path.glob('*'+fp_id+'*.png'):
                fp_file_name = fp_img_path.parts[-1].split('.')[0]
                fp = FCFingerprint(fp_path, fp_file_name)
                self.fps.append(fp)

    def __getitem__(self, index):
        fc_fingerprint = self.fps[index]
        x = fc_fingerprint.image
        gender = fc_fingerprint.gt.gender
        fp_class = fc_fingerprint.gt.fp_class
        fp_class_2 = fc_fingerprint.gt.fp_class_2

        x = torch.from_numpy(x.astype(np.single))
        if self._resize:
            x = T.functional.resize(x, [224, 224])
        x = torch.unsqueeze(x, 0)

        y = np.array(['T', 'A', 'L', 'R', 'W']) == fp_class
        y = torch.from_numpy(y.astype(float))

        if self._hflip and random.random() >= 0.5:
            x = T.functional.hflip(x)

        if self._rotate:
            angle = random.uniform(-20, 20)
            x = T.functional.rotate(x, angle,
                                    interpolation=T.InterpolationMode.BILINEAR)

        return x, y, fp_class_2, gender, index

    def __len__(self):
        return len(self.fps)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_rotate(self, rotate=True):
        self._rotate = rotate

    def set_resize(self, resize=True):
        self._resize = resize


if __name__ == '__main__':
    import sys
    import os
    from argparse import ArgumentParser

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    parser = ArgumentParser(description='FC fingerprint image dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='datasets/fc',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=5, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    num_folds = args.num_folds

    fp_ids = list(range(1, 501))
    random.shuffle(fp_ids)
    splits = np.array(np.array_split(fp_ids, num_folds))

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        fc_fp_ds_val = FCFPDataset(base_path, fp_ids_val)
        fc_fp_ds_tra = FCFPDataset(base_path, fp_ids_tra)

        print('Created FC dataset for validation with {} fingerprints'
              .format(len(fc_fp_ds_val)))
        print(fc_fp_ds_val.fp_ids)

        print('Created FC dataset for training with {} fingerprints'
              .format(len(fc_fp_ds_tra)))
        print(fc_fp_ds_tra.fp_ids)

        print('-' * 80)
