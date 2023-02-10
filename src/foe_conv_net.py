#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn


class FOEConvNet(nn.Module):
    def __init__(self, patch_size, n_classes):
        super(FOEConvNet, self).__init__()

        self.n_classes = n_classes
        self.patch_size = patch_size

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * (patch_size//8)**2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.n_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)


if __name__ == '__main__':
    import os

    from torchvision.transforms import ToTensor

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foe_fingerprint_dataset import FOEFingerprintDataset

    parser = ArgumentParser(description='Fingerprint dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='../datasets/Finger/FOESamples',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-r', '--radius', dest='radius',
                        default=16, type=int, metavar='R',
                        help='radius for patch extraction')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=8, type=int, metavar='Nc',
                        help='number of classes')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    radius = args.radius
    patch_size = args.patch_size
    n_classes = args.n_classes

    fpd_gd = FOEFingerprintDataset(base_path, 'good', 5)
    print('Created a {} fingerprint dataset with {} fingerprints'.
          format(fpd_gd.fp_type, len(fpd_gd)))

    tset_gd, vset_gd = fpd_gd.get_patch_datasets(0, radius, patch_size,
                                                 n_classes)
    patch = tset_gd.patches[0].patch

    x = ToTensor()(patch)
    x = torch.unsqueeze(x, 0)
    model = FOEConvNet(patch_size, n_classes)
    model.eval()
    y = model(x)
    print(x, x.shape)
    print(y, y.shape)
