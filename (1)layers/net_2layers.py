# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:46:43 2019

@author: beach
"""

import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
