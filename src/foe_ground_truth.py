#!/usr/bin/env python3

from pathlib import Path

import numpy as np


def _read_int_bytes(fin, n):
    return int.from_bytes(fin.read(n), byteorder='little')


class FOEGroundTruth:
    def __init__(self, border, step, ori, weights):
        self.border = border
        self.step = step
        self.ori = ori
        self.weights = weights

    @classmethod
    def from_file(cls, base_dir, filename):
        gt_path = Path(base_dir).joinpath(filename)
        with open(gt_path, 'rb') as fin:
            fin.read(8)
            border = _read_int_bytes(fin, 4)
            border_y = _read_int_bytes(fin, 4)
            assert(border == border_y)
            step = _read_int_bytes(fin, 4)
            step_y = _read_int_bytes(fin, 4)
            assert(step == step_y)
            w = _read_int_bytes(fin, 4)
            h = _read_int_bytes(fin, 4)
            ori = np.zeros((h, w), dtype=np.uint8)
            weights = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    ori[r, c] = _read_int_bytes(fin, 1)
                    weights[r, c] = _read_int_bytes(fin, 1)
        return cls(border, step, ori, weights)
