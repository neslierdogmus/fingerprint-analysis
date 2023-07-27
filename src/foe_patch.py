#!/usr/bin/env python3

import numpy as np


class FOEPatch:
    def __init__(self, fp_id, row, column, patch, orientation, fp_type):
        self.fp_id = fp_id
        self.r = row
        self.c = column
        self.patch = np.copy(patch)
        self.ori = orientation
        self.fp_type = fp_type
