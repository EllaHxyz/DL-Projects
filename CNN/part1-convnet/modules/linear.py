"""
Linear Module.  (c) 2021 Georgia Tech

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

import numpy as np


class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        #get the shape of x ->(N,ID)
        N=np.array(x).shape[0]
        ID=1
        for d in x.shape[1:]:
            ID*=d
         
        #X reshape, flatten dimensions -> (N,ID)
        X_flat = np.reshape(x,(N,ID))
        
        #stack of ones for X
        ones = np.ones((N,1)) 
        #X = X add column of 1s -> (N,ID+1), ID is input size
        X_new = np.hstack((X_flat,ones))
        #W = W stack rows of b -> (ID+1, OD), H is hidden_layer_size
        W = np.vstack((self.weight,self.bias))
        
        out = X_new.dot(W) #OUT -> (N,OD)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        N=x.shape[0]
        ID=1
        for d in x.shape[1:]:
            ID*=d
         
        #X reshape, flatten dimensions -> (N,ID)
        X_flat = np.reshape(x,(N,ID))
        
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        #dy/dx = w.T -> (N,ID)
        #reshape (N,ID) -> (N,d1,d2,...)
        dy_dx = dout.dot(self.weight.T)
        self.dx = np.reshape(dy_dx,(x.shape))
        
        #dy/dw = x.T ->(ID,OD)
        self.dw = np.dot(X_flat.T, dout)
        
        #dy/db = 1 -> (OD,)
        #(1,N)x(N,OD) = (1,OD) ->reshape to (OD,)
        self.db = np.ones((1,N)).dot(dout).reshape((-1,))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
