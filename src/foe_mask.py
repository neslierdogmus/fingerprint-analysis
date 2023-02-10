#!/usr/bin/env python3

from pathlib import Path

import numpy as np


class FOEMask:
    def __init__(self, mask_data):
        self.data = np.asarray(mask_data, dtype=np.uint8)

    @classmethod
    def from_file(cls, base_dir, filename):
        mask_path = Path(base_dir).joinpath(filename)
        # with open(mask_path, 'r') as fin:
        #     lines = fin.readlines()
        # h, w = lines[0].strip().split()
        # h, w = int(h), int(w)
        # data = np.zeros((h, w), dtype=np.uint8)
        # for idx, line in enumerate(lines[1:]):
        #     drow = np.asarray([int(item) for item in line.strip().split()],
        #                       dtype=np.uint8)
        #     data[idx, :] = drow

        try:
            data = np.loadtxt(mask_path)
        except ValueError:
            data = np.loadtxt(mask_path, skiprows=1)
        return cls(data)
