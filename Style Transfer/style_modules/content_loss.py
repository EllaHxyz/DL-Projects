import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        content_loss = None
        
        content_loss = content_weight*torch.sum(torch.pow(torch.sub(content_current,content_original),2))
    
        return content_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    
        #_,c,h,w = content_current.shape
        #get F(l) - each row is a feature map channel vectorized, # of row = # of channels
        #torch.reshape(content_current,(1,c,-1)) #->(1,c,m=hxw)
        #get P(l)
        #torch.reshape(content_original,(1,c,-1))
        