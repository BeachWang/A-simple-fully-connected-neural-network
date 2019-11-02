# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:37:51 2019

@author: beach
"""

import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
        nn.init.kaiming_uniform(self.layer1[0].weight.data, mode='fan_in')
        nn.init.kaiming_uniform(self.layer2[0].weight.data, mode='fan_in')
        nn.init.kaiming_uniform(self.layer3[0].weight.data, mode='fan_in')
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x