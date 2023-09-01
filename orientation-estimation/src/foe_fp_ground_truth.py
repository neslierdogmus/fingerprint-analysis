#!/usr/bin/env python3

from pathlib import Path

import numpy as np


def _read_int_bytes(fin, n):
    return int.from_bytes(fin.read(n), byteorder='little')


class FOEGroundTruth:
    def __init__(self, border, step, orientations, weights, mask):
        self.border = border
        self.step = step
        self.orientations = orientations
        self.weights = weights
        self.mask = np.asarray(mask, dtype=np.uint8)

    @classmethod
    def from_file(cls, base_path, fp_id):
        gt_path = Path(base_path).joinpath(fp_id + '.gt')
        mask_path = Path(base_path).joinpath(fp_id + '.fg')

        if not base_path.endswith('Synth'):
            mask = np.loadtxt(mask_path, skiprows=1)
            with open(gt_path, 'rb') as fin:
                fin.read(8)
                border = _read_int_bytes(fin, 4)
                border_y = _read_int_bytes(fin, 4)
                assert (border == border_y)
                step = _read_int_bytes(fin, 4)
                step_y = _read_int_bytes(fin, 4)
                assert (step == step_y)
                w = _read_int_bytes(fin, 4)
                h = _read_int_bytes(fin, 4)
                orientations = np.zeros((h, w))
                weights = np.zeros((h, w), dtype=np.uint8)
                for r in range(h):
                    for c in range(w):
                        orientations[r, c] = _read_int_bytes(fin, 1)
                        orientations[r, c] = orientations[r, c] / 256 * np.pi
                        weights[r, c] = _read_int_bytes(fin, 1)
        else:
            mask = np.loadtxt(mask_path)
            orientations = np.loadtxt(gt_path) / 180 * np.pi
            border = 14
            step = 8
            weights = np.zeros_like(orientations)

        return cls(border, step, orientations, weights, mask)
