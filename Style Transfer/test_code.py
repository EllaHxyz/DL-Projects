#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:56:20 2022

@author: Ella
"""
import torch
import torchvision


tensor1 = torch.randn(2,3)
tensor2 = torch.randn( 2,3, 4,4)
#print(torch.mul(tensor1, tensor2).size())
t1 = torch.unsqueeze(tensor1, dim=-1)
print(t1.size())