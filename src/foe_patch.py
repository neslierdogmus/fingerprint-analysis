#!/usr/bin/env python3

import numpy as np
from foe_orientation import FOEOrientation


class FOEPatch:
    def __init__(self, filename, row, column, patch, orientation, fp_type):
        self.filename = filename
        self.r = row
        self.c = column
        self.patch = np.copy(patch)
        self.ori = FOEOrientation(orientation)
        self.fp_type = fp_type
