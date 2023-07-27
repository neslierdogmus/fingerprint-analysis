#!/usr/bin/env python3
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

from foe_fingerprint import FOEFingerprint


class FOEFPImageDataset(Dataset):
    def __init__(self, base_path, fp_ids, n_classes=1):
        super(FOEFPImageDataset).__init__()
        self.base_path = base_path
        self.fp_ids = fp_ids
        self.n_classes = n_classes
        self._hflip = False
        self.rotate = False
        self.fps = [FOEFingerprint(base_path, fp_id) for fp_id in fp_ids]

    def __getitem__(self, index):
        foe_fingerprint = self.fps[index]
        x = torch.from_numpy(foe_fingerprint.image)/255
        x = torch.unsqueeze(x, 0)

        orientations = foe_fingerprint.gt.orientations

        if self._hflip and random.random() >= 0.5:
            x = tf.hflip(x)
            orientations = np.array([[ori.hflipped() for ori in row]for row in orientations])

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

        return x, y, gt_in_radians, index
        