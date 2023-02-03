#!/usr/bin/env python3
import sys
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Linear(32, self.n_classes),
            nn.Hardtanh(min_val=-1,max_val=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)


if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    from foe_fingerprint import FOEFingerprint

    parser = ArgumentParser(description='Fingerprint dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='/opt/data/FOESamples', metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=32, type=int, metavar='S',
                        help='process fingerprint patches of size SxS')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=8, type=int, metavar='Nc',
                        help='number of classes')
    args = parser.parse_args(sys.argv[1:])

    base_path = Path(args.base_path)
    patch_size = args.patch_size
    n_classes = args.n_classes

    dset = FOEFingerprint.load_index_file(base_path.joinpath('Good'),
                                          'index.txt', True)
    print('Loaded {} fingerprints.'.format(len(dset)))
    random.shuffle(dset)
    print('Randomized fingerprints.')

    patch = dset[0].to_patches(patch_size//2)[0].patch

    x = ToTensor()(patch)
    x = torch.unsqueeze(x, 0)
    print(x, x.shape)
    model = FOEConvNet(patch_size, n_classes)
    y = model(x)
    print(y, y.shape)
