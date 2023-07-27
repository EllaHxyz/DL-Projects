"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        
        #embedding layer
        self.embed = nn.Embedding(output_size, emb_size)
        
        #recurrent layer
        if model_type=="RNN":
            self.recur = nn.RNN(emb_size,decoder_hidden_size,batch_first=True)
        else:
            self.recur = nn.LSTM(emb_size, decoder_hidden_size,batch_first=True)
            
        #linear layer with softmax
        self.linear = nn.Linear(decoder_hidden_size,output_size)
        self.softmax = nn.LogSoftmax()
        
        #dropout layer
        self.dropout = nn.Dropout(dropout)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################

        output, hidden_decoder = None, None
        
        #get embeddings from input
        embed = self.dropout(self.embed(input))
        
        if self.model_type=="RNN":
            out,h_n = self.recur(embed,hidden)
            hidden_decoder = h_n
        else:
            out,(h_n,c_n) = self.recur(embed,hidden)
            hidden_decoder = (h_n,c_n)
        
        #out ->(b,seq,h) to output ->(b,h), h is decoder_hidden_size
        output = out.squeeze(1)
        
        #apply linear layer and softmax; output -> (b,output_size)
        output = self.softmax(self.linear(output))
        
        #hidden ->h_n (1,b,h)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden_decoder
