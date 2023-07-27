#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:01:22 2022

@author: Ella
"""
import numpy as np


y=[[-0.08759809 , 0.21027089 , 0.50813986],
 [-0.98053589,  0.69108355 , 2.36270298],
 [-0.10987781 , 0.21661097,  0.54309974],
 [-1.03143541 , 0.66880383,  2.36904306],
 [-0.18387192 , 0.22847626,  0.64082444],
 [-1.19128892 , 0.59480972 , 2.38090835],
 [-0.2109216  , 0.23004637 , 0.67101435],
 [-1.24695841 , 0.56776003 , 2.38247847]]
y =np.array(y)
print(y.shape)
z=y.reshape((2,2,2,3))
print(z)

arrays = [np.ones((3, 4)) for _ in range(2)]

new = np.stack(arrays, axis=0)
print(new.shape)
out = np.reshape(new,(-1,3*4))
print(out.shape)

k=np.ones((2,2,3,4))
k[:,:,0:1,:]=2
k[:,:,1:2,:]=3
print(k)
print('dim',k.ndim)

mat = k[1,0,:,:]

print('mat',mat)
v={}
model_list = [dict(type='Linear', in_dim=128, out_dim=10),dict(inn=1,out=2),dict(sd=2,un=6)]

for idx,m in enumerate(model_list):
    print(idx)
    print(m)
    v[idx]=dict(dw=np.ones((2,2)), db=np.ones((3,3))*2)
print(v)
print(v[1]['db'])

bc = np.random.randn(32, 128)
print(bc)
def add(a,b):
    a=a+1
    c=a+b
    return c

a=1
b=1
c=add(a,b)
print(a)

w=np.ones((10,3))
w[:,1]=2
w[:,2]=3
print(w)
y=[0,0,0,1,1,1,2,2,2,0]
N=10
out = w[np.arange(N),y]
out = out/2
print(out)
'''
sumk = np.sum(k, axis=(0, 2, 3))
print(sumk)
kt = k.transpose((0,2,3,1)).reshape((-1,2))
print(kt)
'''


'''
rot_k= np.rot90(k, 2, axes=(2, 3))
print(rot_k)
print(k.shape)
print(rot_k.shape)
'''