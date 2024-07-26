#!/usr/bin/env python3

import torch.nn as nn


class FOEConvNet(nn.Module):
    def __init__(self, out_len):
        super(FOEConvNet, self).__init__()

        self.network = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 36, kernel_size=7, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(36, 25, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(25, 25, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(25, 49, kernel_size=9, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(49, 49, kernel_size=9, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(49, 49, kernel_size=9, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(49, 512, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution
            nn.Conv2d(512, out_len, kernel_size=1, stride=1, padding='same'),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.network(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == '__main__':
    import os
    import sys
    import random
    from argparse import ArgumentParser

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from foe_fp_image_dataset import FOEFPImageDataset

    parser = ArgumentParser(description='ConvNet model tests with FOE images')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='./datasets/Finger/FOESamples/Bad',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=1, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=5, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    n_classes = args.n_classes
    num_folds = args.num_folds

    index_path = os.path.join(base_path, 'index.txt')
    with open(index_path, 'r') as fin:
        fp_ids = [line.split('.')[0] for line in fin.readlines()[1:]]
    random.shuffle(fp_ids)
    splits = np.array_split(fp_ids, num_folds)

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        foe_img_ds_val = FOEFPImageDataset([base_path], [fp_ids_val],
                                           n_classes)
        foe_img_ds_tra = FOEFPImageDataset([base_path], [fp_ids_tra],
                                           n_classes)

        x, y = foe_img_ds_val.__getitem__(0)[0:2]
        model = FOEConvNet()
        model.eval()
        y_est = model(x)

        print(x.shape)
        print(y.shape)
        print(y_est.shape)
