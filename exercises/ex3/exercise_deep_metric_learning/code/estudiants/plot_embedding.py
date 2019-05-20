#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:32:37 2018

@author: joans
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(labels, points, images, title='', interactive=False):
    ul = np.unique(labels) # sorts the labels
    dim = points.shape[1]
    assert dim == 2
    symbols = [a+b for a in ['.','o','v','s','x','*','+','p','h']
                       for b in ['r','g','b','c','m','y','k']]*3
    # TODO add more markers and colors so as to have a unique symbol per class
    assert len(symbols) >= len(ul)
    fig = plt.figure()
    for i,lab in enumerate(ul):
        idx = labels==lab
        plt.plot(points[idx,0], points[idx,1], symbols[i], fillstyle='none')

    plt.axis('equal')
    plt.title(title)
    plt.show(block=False)
    
    if interactive:
        print('click at some point and see the image of the nearest identity')
        while True:
            inp = fig.ginput(1)
            p1 = np.array(inp[0])
            plt.figure()
            idx = np.argmin(np.sum((p1-points)**2,axis=1))
            plt.imshow(images[idx], interpolation='nearest')
            plt.axis('off')
            plt.title(labels[idx])
            plt.show(block=False)
    

def plot_tsne(labels, points, lr=100):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    symbols = [a+b for a in ['.','o','v','s','x','*','+','p','h']
                       for b in ['r','g','b','c','m','y','k']]*3
    # TODO add more markers and colors so as to have a unique symbol per class
    ul = np.unique(labels)
    lut = {ul[i] : i for i in range(len(ul))}
    assert len(symbols) >= len(ul)
    
    tsne = TSNE(n_components=2, learning_rate=lr)
    X = tsne.fit_transform(points)
    
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(X[i,0],X[i,1], symbols[lut[labels[i]]])
        
    plt.title('train and test points')

    plt.show(block=False)
    
    

    