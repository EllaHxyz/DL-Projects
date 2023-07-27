"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None
        
    def x_vectorize(self,x,S,K):
         """
         vectorize input (N,IC,H,W) to (H',W',N,IC,K,K)
         """
         x_vector = None
         
         #x shape
         N,IC,H,W = x.shape
         
         #output dimension (H',W')
         H_prime = int((H-self.kernel_size)/self.stride)+1
         W_prime = int((W-self.kernel_size)/self.stride)+1
     
         #construct patches of x,based on kernel size
         patches = []
         for i in range(H_prime):
             for j in range(W_prime):
                 patch = x[:,:,i*S:i*S+K,j*S:j*S+K] #each patch is of shape(N,IC,K,K)
                 patches.append(patch) #-> we have H'*W'number of list (N,IC,K,K)
         
         #stack list to matrix in shape ->(H'*W',N,IC,K,K)
         stack_patches = np.stack(patches,axis=0) 
         #vectorize ->(H',W',N,IC,K,K)
         x_vector = np.array(stack_patches).reshape((H_prime,W_prime,N,IC,K,K))       
                 
                 
         return x_vector
    
    def x_de_vectorize(self,x_vector,N,IC,H,W,H_prime,W_prime,S,K):
        """
        de_vectorize input x_vector = (N,IC,H',W',K,K)->(N,IC,H,W)
        """
        x_dev=None
        
        #construct empty array ->(N,IC,H,W)
        x_dev=np.empty(shape = (N,IC,H,W))
        
        #attach x_vector(N,IC,_,_,K,K) patch to x_dev
        for i in range(H_prime):
            for j in range(W_prime):
                x_dev[:,:,i*S:i*S+K,j*S:j*S+K]=x_vector[:,:,i,j,:,:]
                
        return x_dev
    
    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        S=self.stride
        K=self.kernel_size
        #x_vector-> (H',W',N,IC,K,K)
        x_vector = self.x_vectorize(x, S, K)
        x_sum = np.max(x_vector,axis=(4,5)) #->(H',W'.N,IC)
        
        #(H',W'.N,IC) -> (N,IC,H',W')
        out = x_sum.transpose((2,3,0,1))
        
        #H',W'
        H_out = out.shape[2]
        W_out = out.shape[3]
        #############################################################################
        #      END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, x_vector, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, x_vector, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        H_p,W_p,N,IC,K,L=x_vector.shape
        N,IC,H,W =x.shape
        
        #dout shape ->(N,IC,H',W')
        
        #transpose x_vector (H',W',N,IC,K,K) ->(N,IC,H',W',K,K)
        x_patch = x_vector.transpose((2,3,0,1,4,5))
        
        #construct empty array, same shape as x_patch ->(N,IC,H',W',K,K)
        dy_dx = np.zeros(shape=x_patch.shape)
        
        
        for n,c,h,w in np.ndindex(dout.shape):
            #get gradient value
            gradient = dout[n,c,h,w]
            
            #find the location(indices) of max value in stretched input
            patch = x_patch[n,c,h,w,:,:]
            (j,k) =np.unravel_index(np.argmax(patch),patch.shape)
            
            #use ind(j,k) to allocate dy_dx, gradient flow through that location    
            dy_dx[n,c,h,w,j,k] = gradient
            
        #de-vectorize dy_dx (N,IC,H',W',K,K)->(N,IC,H,W)
        self.dx = self.x_de_vectorize(dy_dx,N,IC,H,W,H_p,W_p,S=self.stride,K=self.kernel_size)
        
        #dy_dx.reshape((N,IC,H_p,W_p,K,K))
        #self.dx = dy_dx.reshape(x.shape)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
