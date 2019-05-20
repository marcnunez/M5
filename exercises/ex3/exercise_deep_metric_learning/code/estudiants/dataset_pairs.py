# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:52:06 2017

@author: joans
"""

import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset

"""
Dataset_of_pairs is a dataset intended to provide minibatches to train a plain 
siamese network. Hence, a batch is *two* sequences of images plus True/False
for each pair of corresponding images, telling whether they are from the same
class or not. 
"""
class Dataset_pairs():
    def __init__(self, dataset, prob_same_class): 
        self.dataset = dataset
        self.prob_same_class = prob_same_class
        self.num_batches = 0


    """ batch_size is the number of pairs of samples """
    def next_batch(self, batch_size):
        first_labels = self.dataset.unique_labels[np.random.randint(low=0, 
            high=self.dataset.num_labels, size=batch_size)]
        num_equal = int(batch_size*self.prob_same_class + 0.5)
        same_label = first_labels[:num_equal]             
        different_label = []
        for i in range(num_equal, batch_size):
            lab = self.dataset.unique_labels[np.random.randint(
                    low=0, high=self.dataset.num_labels)]
            while (lab == first_labels[i]):
                lab = self.dataset.unique_labels[np.random.randint(
                        low=0, high=self.dataset.num_labels)]
            
            different_label.append(lab)
                    
        second_labels = np.hstack([same_label, different_label])           
        y = (np.array(first_labels) != np.array(second_labels)).astype(np.int)
        """ 0 = same class, 1 = different class """

        """ now get a sample for each one of these labels """
        samples1 = []
        samples2 = []
        for lab in first_labels:
            samples1.append(self.dataset.get_random_sample_of_a_class(lab))
        
        for lab in second_labels:
            samples2.append(self.dataset.get_random_sample_of_a_class(lab))
            
        self.num_batches += 1
            
        return Batch_of_pairs(samples1, samples2, y, first_labels, second_labels)

 
"""
Batch_of_pairs just encapsulates the pairs of images and whether each pair is
of one same or different class. This is what Dataset_of_pairs.next_batch(size) 
returns.
"""
class Batch_of_pairs():
    def __init__(self, x1, x2, y, label1=None, label2=None):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        
        """ eventually used just to plot the batch """
        self.label1 = label1
        self.label2 = label2
        
        
    """ shows all the pairs of images in a batch, their label and
    whether they come from the same class or not """
    def show(self):
        batch_size = len(self.x1)
        plt.figure()
        plt.subplot(2,batch_size,1)
        for i in range(batch_size):
            plt.subplot(2,batch_size,i+1)
            plt.imshow(self.x1[i])
            if self.label1 is not None:
                plt.title(str(self.label1[i]),fontsize=8)
            plt.axis('off')
            
            plt.subplot(2,batch_size,batch_size+i+1)
            plt.imshow(self.x2[i])
            if self.label2 is not None:
                title = str(self.label2[i]) + ' ' + str(self.y[i])
            plt.title(title,fontsize=8)
            plt.axis('off')
            
        plt.show(block=False)

        


if __name__ == '__main__':
    dataset_name = 'tsinghua' #'lfw' # 
    if dataset_name == 'lfw':
        path_dataset = '/home/joans/Documents/exercise deep metric learning/datasets/lfwcrop_color_by_dirs/'
        image_extension = 'ppm'
    elif dataset_name == 'tsinghua':
        path_dataset = '/home/joans/Documents/exercise deep metric learning/datasets/tsinghua_resized/'
        image_extension = 'png'
    else:
        assert False

    np.random.seed(0)
    """ to get the same batches at each execution """
    
    ds = Dataset(path=path_dataset, image_extension=image_extension, min_images=20)

    prob_same_class = 0.5
    batch_size = 15
    ds_pairs = Dataset_pairs(ds, prob_same_class)
    batch = ds_pairs.next_batch(batch_size)
    batch.show()
    