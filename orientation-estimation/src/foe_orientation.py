#!/usr/bin/env python3

import numpy as np
import torch


class FOEOrientation:
    def __init__(self, ori_index=0):
        self.theta = ori_index / 256.0 * np.pi

    def _set_radians(self, theta):
        self.theta = theta

    def degrees(self):
        return self.theta / np.pi * 180.0

    def radians(self):
        return self.theta

    def index(self):
        return int(self.theta / np.pi * 256)

    def class_id(self, n_classes):
        return int(self.radians() / np.pi * n_classes)

    def ordinal_code(self, n_classes):
        code_len = n_classes//2
        ind = int(self.radians() / np.pi * n_classes)
        dec = self.radians() / np.pi * n_classes - ind

        if ind < code_len:
            y = torch.zeros(code_len)
            y[:ind] = 1
            y[ind] = dec
        else:
            y = torch.ones(code_len)
            ind = ind - code_len
            y[:ind] = 0
            y[ind] = 1-dec

        return y

    def rotated(self, theta_in_degrees):
        theta = theta_in_degrees * np.pi / 180
        theta_p = self.theta + theta
        if theta_p < 0.0:
            theta_p += np.pi
        elif theta_p >= np.pi:
            theta_p -= np.pi
        return FOEOrientation.from_radians(theta_p)

    def hflipped(self):
        if self.theta == 0:
            return FOEOrientation(0)
        else:
            theta_p = np.pi - self.theta
            return FOEOrientation.from_radians(theta_p)

    @classmethod
    def from_radians(cls, theta):
        new_ori = cls()
        new_ori._set_radians(theta)
        return new_ori

    @staticmethod
    def delta_sqr(angle0_in_radians, angle1_in_radians):
        d = np.fabs(angle0_in_radians - angle1_in_radians)
        if d > np.pi/2.0:
            d = np.pi - d
        return d**2

    @staticmethod
    def radians_from_mode(distribution):
        n_classes = len(distribution)
        class_id = np.argmax(distribution)
        radians = np.pi / n_classes * (class_id + 0.5)
        return radians

    @staticmethod
    def radians_from_interpolated_mode(distribution):
        peak_id = np.argmax(distribution).item()
        N = len(distribution)
        pm = peak_id - 1
        if pm < 0:
            pm = N - 1
        pp = peak_id + 1
        if pp == N:
            pp = 0
        yp = distribution[pp].item()
        y0 = distribution[peak_id].item()
        ym = distribution[pm].item()
        peak_x = (ym - yp) / (2*yp + 2*ym - 4*y0)
        return (peak_id + peak_x + 0.5) * np.pi / N

    @staticmethod
    def radians_from_marginalization(distribution):
        N = len(distribution)
        cx, sx = 0.0, 0.0
        for idx, p in enumerate(distribution):
            x = (idx+0.5) * np.pi / N * 2.0
            cx += p.item() * np.cos(x)
            sx += p.item() * np.sin(x)
        cx /= N
        sx /= N
        theta = np.arctan2(sx, cx)
        if theta < 0.0:
            theta += 2.0 * np.pi
        return theta / 2.0

    @staticmethod
    def radians_from_sincos(sincos):
        radians = np.arctan(sincos[:, 0] / sincos[:, 1])
        radians[radians < 0] = radians[radians < 0] + np.pi
        return radians

    @staticmethod
    def radians_from_ordinal_code(codes):
        def ret_func(code, th=0):
            code_len = len(code)
            code = torch.tensor(code)
            n_classes = code_len*2
            dist_fnc = torch.nn.functional.binary_cross_entropy_with_logits

            # dist1 = [torch.nn.functional.mse_loss(code, c)
            #          for c in codes]
            # dist2 = [dist_fnc(code, c) for c in codes]
            dist2 = dist_fnc(code.repeat(n_classes, 1),
                             codes, reduction='none').mean(axis=1)

            # m = torch.nn.Softmax()
            # prob1 = m(-1*torch.tensor(dist1))
            # prob2 = m(-1*torch.tensor(dist2))

            # ret1 = np.argmin(dist1) / n_classes * np.pi
            ret2 = np.argmin(dist2) / n_classes * np.pi
            # ret3 = FOEOrientation.radians_from_marginalization(prob1)
            # ret4 = FOEOrientation.radians_from_marginalization(prob2)

            return ret2

        return ret_func

    @staticmethod
    def create_codes_list(num_classes):
        codes = []
        for i in range(num_classes):
            theta_p = np.pi / num_classes * i
            ori = FOEOrientation.from_radians(theta_p)
            codes.append(ori.ordinal_code(num_classes))
        return torch.stack(codes)
