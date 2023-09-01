#!/usr/bin/env python3

import numpy as np


class FMDGroundTruth:
    def __init__(self, width, height, quality, lst_min_type, lst_min_posX,
                 lst_min_posY, lst_min_angle, lst_min_qual):
        self.width = width
        self.height = height
        self.quality = quality

        self.lst_min_type = lst_min_type
        self.lst_min_posX = lst_min_posX
        self.lst_min_posY = lst_min_posY
        self.lst_min_angle = lst_min_angle
        self.lst_min_qual = lst_min_qual

    @classmethod
    def from_file(cls, gt_path):
        npz_file = np.load(gt_path)
        width = npz_file['size_x']
        height = npz_file['size_y']
        quality = npz_file['finger_qual']

        lst_min_type = npz_file['T']
        lst_min_posX = npz_file['X']
        lst_min_posY = npz_file['Y']
        lst_min_angle = npz_file['A']
        lst_min_qual = npz_file['Q']

        return cls(width, height, quality, lst_min_type, lst_min_posX,
                   lst_min_posY, lst_min_angle, lst_min_qual)
