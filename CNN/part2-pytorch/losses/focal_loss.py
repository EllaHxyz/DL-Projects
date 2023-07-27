"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

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
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    #get alpha weight for each cls
    alpha = [] 
    
    for i in range(len(cls_num_list)):
        n_i=cls_num_list[i]
        #get Eni for each cls
        E_n_i=(1-beta**n_i)/(1-beta)
    
        #get alpha for each cls
        alpha.append(1/E_n_i)
    
    #normalize alpha by total number of cls (C)
    C=len(cls_num_list)+1
    alpha = np.array(alpha)
    alpha = alpha/np.sum(alpha)*C
            
    #assign
    per_cls_weights = alpha
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        
        N, _ = input.shape
       
        #METHOD #3
        #get alpha weights for each data point
        alpha = []
        for tg in target:
            alpha.append(self.weight[tg])
        
        alpha = torch.tensor(alpha)
        
        #CE-SOFTMAX LOSS, with weight, no reduction
        CE = F.cross_entropy(input,target,reduction='none')
        
        #calculate pt, based on l = -log(pt) for each data point
        pt = torch.exp(-CE)
        
        #Focal Loss = alpha weight * Focal term (1-pt)^gamma * CE LOSS
        FL_CE = alpha*(1-pt)**self.gamma*CE
            
        loss = FL_CE.sum()
        
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
    
    
        '''
        #METHOD #1
        #softmax probs
        m = nn.Softmax(dim=1)
        probs = m(input)
        #get ground truth prob
        pt = probs[torch.arange(N),target]
        #Focal loss for each data point
        FL = -(1-pt)**self.gamma*torch.log(pt)
        
       
        #get alpha weights for each data point
        alpha = []
        for tg in target:
            alpha.append(self.weight[tg])
        
        alpha = torch.tensor(alpha)
        
        #class balanced-focal loss: CB-FL = FL*Effective sample weights
        loss = torch.sum(alpha*FL)/N
        '''
        
        
        '''
       # METHOD #2
        #get alpha weights for each data point
        alpha = []
        for tg in target:
            alpha.append(self.weight[tg])
        
        alpha = torch.tensor(alpha)
        
        #weight for each class, size = C
        w_per_cls = torch.tensor(self.weight).float()
        
        #CE-SOFTMAX LOSS, with weight, no reduction
        CE = F.cross_entropy(input,target,weight=w_per_cls,reduction='none')
        
        #calculate pt, based on l = -wlog(pt) for each data point
        pt = torch.exp(-CE/alpha)
        
        #Focal Loss = Focal term (1-pt)^gamma * CE LOSS
        FL_CE = (1-pt)**self.gamma*CE
            
        #print('CE',CE[:5])
        #print('PT',pt[:5])
        loss = FL_CE.sum()
        
        '''
