#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from fmd_fingerprint import FMDFingerprint


class FMDFPImageDataset(Dataset):
    def __init__(self, base_path, fp_ids, n_classes=1):
        super(FMDFPImageDataset).__init__()

        self.base_path = base_path
        self.fp_ids = fp_ids
        self.n_classes = n_classes
        self._hflip = False
        self._rotate = False
        self._resize = False
        self.fps = []

        folders = ['FM3_FVC2002DB1A_np', 'FM3_FVC2002DB3A_np',
                   'FM3_FVC2004DB1A_np', 'FM3_FVC2004DB3A_np']
        for fp_id in fp_ids:
            fp_folder = folders[fp_id // 100]
            fp_path = Path(base_path).joinpath(fp_folder)
            fp_fid = str(fp_id % 100 + 1) + '_'
            for fp_img_path in fp_path.glob(fp_fid+'*.tif'):
                fp_fname = fp_img_path.parts[-1].split('.')[0]
                fp = FMDFingerprint(fp_path, fp_fname)
                if (not isinstance(fp.image, type(None))
                        and len(fp.gt.lst_min_type) > 0):
                    self.fps.append(fp)

    def __getitem__(self, index):
        fmd_fingerprint = self.fps[index]
        x = fmd_fingerprint.image
        y = fmd_fingerprint.get_minutiae_map()
        # y_b = fmd_fingerprint.get_minutiae_map(3,0.00001)
        quality = fmd_fingerprint.gt.quality
        min_x = fmd_fingerprint.gt.lst_min_posX
        min_y = fmd_fingerprint.gt.lst_min_posY
        x = torch.from_numpy(x.astype(np.single))
        y = torch.from_numpy(y.astype(np.single))
        # y_b = torch.from_numpy(y_b.astype(np.single))
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        # y_b = torch.unsqueeze(y_b, 0)

        if self._hflip and random.random() >= 0.5:
            x = T.functional.hflip(x)
            y = T.functional.hflip(y)
            # y_b = T.functional.hflip(y_b)
            min_x = x.shape[2] - min_x

        if self._rotate:
            angle = random.uniform(-20, 20)
            int_type = T.InterpolationMode.NEAREST
            x = T.functional.rotate(x, angle, interpolation=int_type)
            y = T.functional.rotate(y, angle, interpolation=int_type)
            # y_b = T.functional.rotate(y_b, angle, interpolation=int_type)
            yc, xc = x.shape[1]/2, x.shape[2]/2
            angle_r = np.deg2rad(-angle)
            min_xx = np.cos(angle_r)*(min_x-xc) - np.sin(angle_r)*(min_y-yc) + xc
            min_yy = np.sin(angle_r)*(min_x-xc) + np.cos(angle_r)*(min_y-yc) + yc
            min_x = np.round(min_xx).astype(int)
            min_y = np.round(min_yy).astype(int)
        
        if self._resize:
            scale = random.uniform(0.8, 1.2)
            H = int(x.shape[1] * scale)
            W = int(x.shape[2] * scale)
            x = T.functional.resize(x, [H, W], antialias=True)
            y = T.functional.resize(y, [H, W], antialias=True)
            # y_b = T.functional.resize(y_b, [H, W], antialias=True)
            min_x = np.round(min_x * scale).astype(int)
            min_y = np.round(min_y * scale).astype(int)

        valid_ind = np.all((0 <= min_x, min_x < x.shape[2],
                            0 <= min_y, min_y < x.shape[1]), axis=0)
        min_x = min_x[valid_ind]
        min_y = min_y[valid_ind]
        y_b = np.zeros_like(x)
        y_b[0, min_y, min_x] = 1
        y_b = torch.from_numpy(y_b.astype(np.single))
        y_b = torch.unsqueeze(y_b, 0)
        x = T.functional.center_crop(x, [480, 640])
        y = T.functional.center_crop(y, [480, 640])
        y_b = T.functional.center_crop(y_b, [480, 640])

        # if list(x.shape) == [1, 374, 388]:
        #     x = T.functional.pad(x, (126, 53))
        #     y = T.functional.pad(y, (126, 53))
        #     y_b = T.functional.pad(y_b, (126, 53))
        #     # min_x = min_x + 126
        #     # min_y = min_y + 53
        # elif list(x.shape) == [1, 300, 300]:
        #     x = T.functional.pad(x, (170, 90))
        #     y = T.functional.pad(y, (170, 90))
        #     y_b = T.functional.pad(y_b, (170, 90))
        #     # min_x = min_x + 170
        #     # min_y = min_y + 90
        # elif list(x.shape) == [1, 480, 300]:
        #     x = T.functional.pad(x, (170, 0))
        #     y = T.functional.pad(y, (170, 0))
        #     y_b = T.functional.pad(y_b, (170, 0))
        #     # min_x = min_x + 170

        min_y = torch.nonzero(y_b)[:,2].numpy()
        min_x = torch.nonzero(y_b)[:,3].numpy()
        min_y = np.pad(min_y, (0, 100-len(min_y)), constant_values=(-1))
        min_x = np.pad(min_x, (0, 100-len(min_x)), constant_values=(-1))

        x = T.functional.normalize(x, torch.mean(x), torch.std(x))

        return x, y, (min_y, min_x), quality, index

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

    parser = ArgumentParser(description='FMD fingerprint image dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='datasets/fmd',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=5, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    num_folds = args.num_folds

    fp_ids = list(range(400))
    random.shuffle(fp_ids)
    splits = np.array(np.array_split(fp_ids, num_folds))

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        fc_fp_ds_val = FMDFPImageDataset(base_path, fp_ids_val)
        fc_fp_ds_tra = FMDFPImageDataset(base_path, fp_ids_tra)

        print('Created FC dataset for validation with {} fingerprints'
              .format(len(fc_fp_ds_val)))
        print(fc_fp_ds_val.fp_ids)

        print('Created FC dataset for training with {} fingerprints'
              .format(len(fc_fp_ds_tra)))
        print(fc_fp_ds_tra.fp_ids)

        print('-' * 80)
