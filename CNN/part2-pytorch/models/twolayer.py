"""
Two Layer Network Model.  (c) 2021 Georgia Tech

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

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.lin1 = nn.Linear(in_features = input_dim,out_features = hidden_size)
        self.lin2 = nn.Linear(in_features = hidden_size,out_features= num_classes)
        #self.ic = input_dim
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        '''
        x = x.reshape((-1,self.ic))
        x = self.lin1(x)
        print(x.shape)
        x = F.sigmoid(x)
        print(x.shape)
        out = self.lin2(x)
        print(out.shape)
        '''
        #x = x.reshape((-1,self.ic))
        N=x.shape[0]
        x=x.reshape((N,-1))
        out = self.lin2(F.sigmoid(self.lin1(x)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
