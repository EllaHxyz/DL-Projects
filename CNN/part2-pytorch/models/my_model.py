"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        '''
        conv 1-> relu/max pool -> conv2 ->relu/max pool->fcl1 ->relu ->fcl2
        '''
        '''
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=12, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels = 12,out_channels=64, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fcl1 = nn.Linear(in_features = 64*5*5,out_features = 60)
        #self.fcl2 = nn.Linear(in_features = 60,out_features=20)
        self.fcl2 = nn.Linear(in_features =60,out_features=10)
        '''
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=12, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels = 12,out_channels=64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=512,kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fcl1 = nn.Linear(in_features = 512*2*2,out_features = 60)
        self.fcl1_bn = nn.BatchNorm1d(60)
        #self.fcl2 = nn.Linear(in_features = 60,out_features=20)
        self.fcl2 = nn.Linear(in_features =60,out_features=10)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        '''
        N=x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape((N,-1)) #x = torch.flatten(x,1)
        out = self.fcl2(self.fcl1(x))
        '''
        N=x.shape[0]
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.reshape((N,-1)) #x = torch.flatten(x,1)
        out = self.fcl2(self.fcl1_bn(self.fcl1(x)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
