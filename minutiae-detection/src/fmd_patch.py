#!/usr/bin/env python3

import numpy as np


class FMDPatch:
    def __init__(self, fp_fname, row, column, patch, orientation, fp_type):
        self.fp_fname = fp_fname
        self.r = row
        self.c = column
        self.patch = np.copy(patch)
        # self.ori = orientation!!
        self.fp_type = fp_type
