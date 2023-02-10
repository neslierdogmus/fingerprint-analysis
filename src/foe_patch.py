#!/usr/bin/env python3

import numpy as np


class FOEOrientation:
    def __init__(self, ori_index):
        self.index = ori_index

    def degrees(self):
        return self.index / 256.0 * 180.0

    def radians(self):
        return self.index / 256.0 * np.pi

    def class_id(self, n_classes):
        return int(self.index / 256.0 * n_classes)

    def rotated(self, theta_in_radians):
        theta_p = self.radians() + theta_in_radians
        if theta_p < 0.0:
            theta_p += np.pi
        elif theta_p >= np.pi:
            theta_p -= np.pi
        return FOEOrientation.from_radians(theta_p)

    def hflipped(self):
        if self.index == 0:
            return FOEOrientation(0)
        else:
            theta_p = np.pi - self.radians()
            return FOEOrientation.from_radians(theta_p)

    @classmethod
    def from_radians(cls, theta):
        ori_index = int(theta / np.pi * 256.0)
        return cls(ori_index)

    @staticmethod
    def delta(angle0_in_radians, angle1_in_radians):
        d = np.fabs(angle0_in_radians - angle1_in_radians)
        if d > np.pi/2.0:
            d = np.pi - d
        return d

    @staticmethod
    def radians_from_modes(distributions):
        n_classes = distributions.shape[1]
        class_ids = np.argmax(distributions, 1)
        radians = np.pi / n_classes * (class_ids + 0.5)
        return radians

    @staticmethod
#    def estimation_error_sqr(gt_in_radians, distributions):
    def estimation_error_sqr(gt_in_radians, est_in_radians):
#         est_in_radians = FOEOrientation.radians_from_modes(distributions)
        deltas = gt_in_radians - est_in_radians
        deltas[deltas > np.pi/2.0] = np.pi - deltas[deltas > np.pi/2.0]
        delta_sqr = deltas ** 2
#        return delta_sqr
        return delta_sqr.sum()


class FOEPatch:
    def __init__(self, filename, row, column, patch, orientation, is_good):
        self.filename = filename
        self.r = row
        self.c = column
        self.patch = np.copy(patch)
        self.ori = FOEOrientation(orientation)
        self.is_good = is_good
