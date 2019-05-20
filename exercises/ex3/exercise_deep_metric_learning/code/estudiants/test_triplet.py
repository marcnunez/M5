# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:06:21 2018

@author: joans
"""
import numpy as np
import tensorflow as tf
import os


if False:
    y = [1,1,1,2,2,3]
    x = np.array([[1,1,1,1],
                  [2,2,2,2],
                  [3,3,3,3],
                  [4,4,4,4],
                  [5,5,5,5],
                  [6,6,6,6],
                 ])
    nsamples, dim = x.shape

else:
    P = 3 # classes or different labels
    K = 2 # images/class
    dim = 4 # dimension of features
    # K(K-1)(P-1)K triples with anchor of the same class
    # (P-1)K triplets with one same anchor -> needed for batch hard mining

    np.random.seed(0)
    perm = np.random.permutation(10)
    assert P*K<=10
    x = np.array([ [perm[i]]*dim for i in range(P*K)])
    y = list(np.ravel([[i]*K for i in range(P)]))
    nsamples = P*K
    assert nsamples==len(x)
    assert len(x)==len(y)



gpu = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


emb = tf.placeholder(tf.float32, [None,dim])
labels = tf.placeholder(tf.int32, [None])

lab1, lab2, lab3 = tf.meshgrid(labels, labels, labels)
""" cartesian product """
idx = tf.where(tf.logical_and(tf.equal(lab1,lab2), tf.not_equal(lab1,lab3)))
""" anchor and positive with same class (label), negative different class """
idx1, idx2, _ = tf.split(idx, axis=1, num_or_size_splits=3)
idx1 = tf.squeeze(idx1)
idx2 = tf.squeeze(idx2)
idx_dif = tf.squeeze(tf.gather(idx, tf.where(tf.not_equal(idx1,idx2))))
""" anchor different from positive """
sel_labels = tf.gather(labels,idx_dif)
sel_emb = tf.gather(emb,idx_dif)
s1, s2, s3 = tf.split(sel_emb, axis=1, num_or_size_splits=3)
anchors = tf.squeeze(s1)
positives = tf.squeeze(s2)
negatives = tf.squeeze(s3)



sess = tf.InteractiveSession()

print('\ntest of cartesian product labels x labels x labels, labels=[1,2,3]')
l = [1,2,3]
num_labels = len(l)

l1=lab1.eval(feed_dict={labels:l})
l2=lab2.eval(feed_dict={labels:l})
l3=lab3.eval(feed_dict={labels:l})

count = 1
for i in range(num_labels):
  for j in range(num_labels):
    for k in range(num_labels):
      print([count, [l1[i,j,k],l2[i,j,k],l3[i,j,k]]])
      count += 1

a, p, n, l1, l2, l3, index_different, selected_labels = \
    sess.run([anchors, positives, negatives, lab1, lab2, lab3, idx_dif, sel_labels],
              feed_dict={emb:x, labels:y})


print('\ntest of valid triplets')
print('x')
print(x)
print('y')
print(y)

if False:
  num_labels = len(y)
  count = 1
  for i in range(num_labels):
    for j in range(num_labels):
      for k in range(num_labels):
        print([count, [l1[i,j,k],l2[i,j,k],l3[i,j,k]]])
        count += 1


num_triplets = len(a)
for i in range(num_triplets):
    print('triplet {}, labels {}, \ta={} \tp={} \tn={}'.format(i,selected_labels[i],a[i],p[i],n[i]))


# batch hard : for each anchor find the distance from anchor to the hardest positive
# (farthest away) and to the hardest negative (closer) and then apply the regular
# triplet loss function [m + d_ap - d_an]_+  => only PK terms
val_P = tf.constant(P, tf.int32)
val_K = tf.constant(K, tf.int32)

d2_ap = tf.reduce_sum( tf.pow(anchors - positives, 2), 1)
d_ap = tf.sqrt(d2_ap + 1e-8, name="eucd_ap")
d2_an = tf.reduce_sum( tf.pow(anchors - negatives, 2), 1)
d_an = tf.sqrt(d2_an + 1e-8, name="eucd_an")
margin = tf.constant(1.0, dtype=tf.float32, name="margin")
triplet_losses = tf.maximum(0.0, margin + d_ap - d_an)
loss_tri = tf.reduce_mean(triplet_losses, name='triplet_loss')

d_ap_by_a = tf.reshape(d_ap,[-1, (val_P-1)*val_K])
d_an_by_a = tf.reshape(d_an,[-1, (val_P-1)*val_K])
d_hard_pos = tf.reduce_max(d_ap_by_a,axis=1)
d_hard_neg = tf.reduce_min(d_an_by_a,axis=1)

batch_hard_losses = tf.maximum(0.0, margin + d_hard_pos - d_hard_neg)
loss_bh = tf.reduce_mean(batch_hard_losses)

for z in [d_ap_by_a, d_an_by_a, d_hard_pos, d_hard_neg,
          batch_hard_losses, loss_bh]:
    print(z.eval(feed_dict={emb:x, labels:y, val_P:P, val_K:K}))


