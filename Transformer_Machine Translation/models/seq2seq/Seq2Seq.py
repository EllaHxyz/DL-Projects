import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

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
import torch.optim as optim
import numpy as np

# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder=encoder
        self.decoder=decoder
        self.model_type = decoder.model_type
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        
        outputs = None
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder.                           #
        # encoder -> context vector
        if self.model_type=="RNN":      
            _, hidden = self.encoder(source)
        else:
            out,(h_n,c_n) = self.encoder(source)
            hidden = (h_n,c_n)
            
        
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence. 
        
        #source(b,seq) -> input(b,1)   
        #indices (b,1)
        indices = torch.zeros(batch_size,1,dtype=int) 
        input = torch.gather(source, 1, indices)
        
        
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs. 
        
        #init outputs                                                  
        outputs = None
        
        for t in range(out_seq_len):
            #output ->(b,out_size)
            #outputs -> (seq,b,out_size)
            if self.model_type=="RNN":
                output, h_n = self.decoder(input,hidden)
                #update hidden fed into decoder at each time step
                hidden = h_n
            else:
                output, (h_n,c_n) = self.decoder(input,hidden)
                #update hidden
                hidden= (h_n,c_n)
            
            
            #add output to final outputs tensor: (seq+1,b,out)
            if outputs is None:
                #(b,out)->(1,b,out)
                outputs = output.unsqueeze(0)    
            else:
                #concate: (seq,b,out) +(1,b,out)->(seq+1,b,out)
                outputs = torch.cat((outputs,output.unsqueeze(0)),dim=0)
                
            
            # manipulate output: choose the highest softmax scores to generate the pred for next input
            #input - >(b,1); output:(b,out_size)->(b,1)
            
            #max_index:(b,out_size)->(b,)->(b,1) 
            max_index = torch.argmax(output,dim=1).unsqueeze(-1)
            
            #update input fed into decoder at each time step
            input = max_index
              
        
        #manipulates outputs dim (seq,b,out_size)->(b,seq,out_size)
        outputs = torch.transpose(outputs,0,1)

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
