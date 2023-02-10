#!/usr/bin/env python3
import torch
import torch.nn as nn


class FOE_CNN(nn.Module):
    def __init__(self, path, inp_dim, out_dim, device):
        super().__init__()

        self.path = path
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.device = device

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
            # nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        self.linear_layers = torch.nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * (inp_dim//8)**2, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.out_dim//2),
            # nn.ReLU(inplace=True)
        )

        self.apply(self.init_weights)
        self = self.to(device)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight,
                                          gain=nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
