# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:23:45 2018

@author: joans
"""

import numpy as np
import tensorflow as tf
import os

if False:
    x = np.array([[1,1,1,1],
                  [2,2,2,2],
                  [3,3,3,3],
                  [4,4,4,4],
                 ])
    y = [1,1,2,2]
    nsamples, dim = x.shape
else:
    P = 5 # classes or different labels
    K = 3 # images/class
    dim = 4 # dimension of features
    np.random.seed(0)
    val = np.random.randint(low=0,high=10,size=P*K)
    #assert P*K<=10
    x = np.array([ [val[i]]*dim for i in range(P*K)])
    y = list(np.ravel([[i]*K for i in range(P)]))
    nsamples = P*K
    assert nsamples==len(x)
    assert len(x)==len(y)



gpu = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

emb = tf.placeholder(tf.float32, [None,dim])
labels = tf.placeholder(tf.int32, [None])

sess = tf.InteractiveSession()

r = tf.range(tf.shape(emb)[0])
i, j = tf.meshgrid(r, r)
""" cartesian product """
idx = tf.where(tf.greater(i,j))
out1, out2 = tf.split(tf.gather(x,idx),axis=1,num_or_size_splits=2)
lab1, lab2 = tf.split(tf.gather(labels,idx),axis=1,num_or_size_splits=2)
out1 = tf.squeeze(out1)
out2 = tf.squeeze(out2)
lab1 = tf.squeeze(lab1)
lab2 = tf.squeeze(lab2)
same_dif = 1.0 - tf.to_float(tf.equal(lab1, lab2))
""" 1 - tf.to_... because the convention in the contrastive loss is that
1 = different and 0 = same, but float(True)==1 """

sess = tf.InteractiveSession()

ei, ej, eidx, eout1, eout2, elab1, elab2, esame_dif = \
  sess.run([i, j, idx, out1, out2, lab1, lab2, same_dif], 
           feed_dict={emb:x, labels:y})

print('\ntest contrastive')
print('x')
print(x)
print('y')
print(y)

for i in range(len(eout1)):
  print(eout1[i], eout2[i], elab1[i], elab2[i], esame_dif[i])

num_same = np.sum(esame_dif==0)
num_dif = np.sum(esame_dif==1)
print('for P={}, K={}, {} pairs same class and {} pairs different class, {}%'.\
  format(P, K, num_same, num_dif, np.round((100.*num_same)/num_dif, decimals=2)))  
