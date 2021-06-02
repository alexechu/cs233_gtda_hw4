"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, init_feat_dim=3, n_points=1024, pooldim=1024, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ reLUs and the (max) poolings.
        """
        super(PointNet, self).__init__()
        self.layers = []
        self.conv_dims = [init_feat_dim] + conv_dims
        for i in range(len(self.conv_dims)-1):
            # self.layers.append(nn.Conv1d(self.conv_dims[i], self.conv_dims[i+1], kernel=(1, 3), stride=1, padding=0, bias=True))
            # k = 3 if i == 0 else 1
            self.layers.append(nn.Conv1d(self.conv_dims[i], self.conv_dims[i+1], 1, 1, 0, bias=True))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers = nn.ModuleList(self.layers)
        self.pool = nn.MaxPool1d(pooldim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        for layer in self.layers:
            x = layer(x)
        # x = x.transpose(-1, -2)
        # print('prepool', x.shape)
        return self.pool(x)
