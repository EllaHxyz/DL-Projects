"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None
    
    def x_vectorize(self,x,P,S,K):
        """
        vectorize input (N,IC,H,W) to (N*H'*W',IC*K*K)
        """
        x_vector = None
        
        N,IC,H,W = x.shape
        
        #output dimension (H',W')
        H_prime = int((H+2*self.padding-self.kernel_size)/self.stride)+1
        W_prime = int((W+2*self.padding-self.kernel_size)/self.stride)+1
    
        #padding x (N,IC,H,W)->(N,IC,H+2P,W+2P)
        x_pad = np.pad(x,((0,0),(0,0),(P,P),(P,P)),'constant',constant_values=(0,0))
        
        #construct patches of x,based on kernel size
        patches = []
        for i in range(H_prime):
            for j in range(W_prime):
                patch = x_pad[:,:,i*S:i*S+K,j*S:j*S+K] #each patch is of shape(N,IC,K,K)
                patches.append(patch) #-> we have H'*W'number of list (N,IC,K,K)
        
        #stack list to matrix in shape ->(H'*W',N,IC,K,K)
        stack_patches = np.stack(patches,axis=0) 
        #vectorize ->(H',W',N,IC,K,K)
        x_vector = np.array(stack_patches).reshape((H_prime,W_prime,N,IC,K,K))       
                
                
        return x_vector
    
    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        '''Get Dimensions'''
        #input dimension (H,W)
        N,IC,H,W = x.shape
        
        #output dimension (H',W')
        H_prime = int((H+2*self.padding-self.kernel_size)/self.stride)+1
        W_prime = int((W+2*self.padding-self.kernel_size)/self.stride)+1
        
        #P,S,K
        P=self.padding
        S=self.stride
        K=self.kernel_size
        
        '''Vectorize W and X'''
        #vectorize W (OC,IC,K,K) ->(IC*K*K,OC)
        W=self.weight.reshape((self.out_channels, self.in_channels*self.kernel_size*self.kernel_size)).T
        
        #vectorize X (N,IC,K,K) -> (H',W',N,IC,K,K) ->(H'*W'*N,IC*K*K)
        x_vector = self.x_vectorize(x, P, S, K)
        #flatten x ->(H'*W'*N,IC*K*K)
        X=x_vector.reshape((H_prime*W_prime*N,IC*K*K)) 
        
        '''Add bias'''
        #add bias: X stack col of ones + W stack rows of b
        X_new = np.hstack((X,np.ones((N*H_prime*W_prime,1)))) #->(,+1)
        W_new = np.vstack((W,self.bias))
        
        #output ->(H'*W'*N,OC)
        Y = np.dot(X_new,W_new)
        
        #reshape to (H',W',N,OC)
        Y_reshape = Y.reshape((H_prime,W_prime,N,self.out_channels))
        
        #transpose dimensions to(N,OC,H',W')
        out = Y_reshape.transpose(2,3,0,1)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x, x_vector
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x,x_vector= self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        '''
        #dout (N,OC,H',W') ->(N*H'*W',OC)
        D = dout.transpose((0,2,3,1)).reshape((-1,self.out_channels))
        dy_dw = X.T.dot(D).reshape((self.in_channels,self.kernel_size,self.kernel_size,self.out_channels))
        self.dw = dy_dw.transpose((3,0,1,2)) #(IC,K,K,OC) TO (OC,IC,K,K)
        '''
        #rotate W upside-down -> shape unchanged (OC,IC,K,L)
        w_rotate = np.rot90(self.weight, 2, axes=(2, 3))
        #dout (N,OC,H',W') to (H'',W'',N,OC,K,L)
        d_vector = self.x_vectorize(dout,P=self.padding,S=1,K=self.kernel_size)
        
        #dx = dout_vector x w_rotate
        #d_vector ->(H',W',N,OC,K,L); w_rotate ->(OC,IC,K,L)
        self.dx = np.einsum('hwnokl,oikl->nihw',d_vector,w_rotate)
        
        #dw = x_vector x dout
        self.dw = np.einsum('hwnikl,nohw->oikl', x_vector, dout)
        
        #db = (1,N*H'*W')x(N*H'*W',OC) ->(OC,)
        N,OC,H_prime,W_prime = dout.shape
        self.db = np.ones((1,N*H_prime*W_prime)).dot(dout.transpose((0,2,3,1)).reshape((-1,OC))).reshape((-1,))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
