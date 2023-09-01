#!/usr/bin/env python3


class FCGroundTruth:
    def __init__(self, gender, fp_class, fp_class_2):
        self.gender = gender
        self.fp_class = fp_class
        self.fp_class_2 = fp_class_2

    @classmethod
    def from_file(cls, gt_path):
        with open(gt_path) as fin:
            try:
                gender = fin.readline().split()[1]
            except IndexError:
                gender = ''
            fp_class = fin.readline().split()[1]
            try:
                fp_class_2 = fin.readline().split()[2][1]
            except IndexError:
                fp_class_2 = ''

        return cls(gender, fp_class, fp_class_2)
