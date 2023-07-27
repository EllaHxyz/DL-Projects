import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # HINT: you may find torch.bmm() function is handy when it comes to process  #
        # matrix product in a batch. Please check the document about how to use it.  #
        ##############################################################################
        
        n,c,h,w = features.shape
        
        #get F(l) - each row is a feature map channel vectorized, # of row = # of channels
        #F = torch.reshape(features,(n,c,-1)) #->(n,c,m=hxw)
        F = torch.flatten(features,start_dim = 2,end_dim=3)
        
        #get F'(l) -(N,M,C)
        F_t = torch.transpose(F,1,2)
        
        #mat batch multiplication (n,c,m) x (n,m,c) ->(N,C,C)
        gram = torch.bmm(F,F_t)
        
        #normalize, divide by number of neurons
        if normalize:
            gram = gram/(h*w*c)
        
        return gram
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of gram_matrix of each style layers, the same length as style_layers, 
              where style_targets[i] is a PyTorch Variable giving the Gram matrix of the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """

        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################
        
        
        style_loss = 0
        
        for i in range(len(style_layers)):
              
            #get features of a particular layer in current image
            layer = style_layers[i]
            features = feats[layer]
            
            #compute gram of current feature layer
            gram = self.gram_matrix(features)
            #get gram of target feature layer
            gram_target = style_targets[i]
            
            #style loss of a particular layer
            style_loss += style_weights[i]*torch.sum(torch.pow(torch.sub(gram,gram_target),2))
        
        return style_loss
            
            
            
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

