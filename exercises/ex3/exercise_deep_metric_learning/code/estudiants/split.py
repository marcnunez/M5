# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:45:15 2018

@author: joans
"""
import os
import numpy as np

from dataset import Dataset


class Split():
    def __init__(self, dataset, ratios):
        assert np.abs(1.0 - np.sum(list(ratios.values()))) < 1e-6
        self.ratios = ratios
        """ ratio of samples in train, validation and test datasets into which
        dataset is split """
        
        self.filenames_train = {}
        self.filenames_valid = {}
        self.filenames_test = {}
        
        labels = dataset.unique_labels
        for lab in labels:
            num_samples_train = int(len(dataset.filenames[lab])*self.ratios['train'])
            num_samples_valid = int(len(dataset.filenames[lab])*self.ratios['valid'])
            
            fn = dataset.filenames[lab][:num_samples_train]
            self.filenames_train.update({lab : fn})
            
            idx = num_samples_train
            fn = dataset.filenames[lab][idx:idx+num_samples_valid]
            self.filenames_valid.update({lab : fn})
            
            idx += num_samples_valid
            fn = dataset.filenames[lab][idx:]
            self.filenames_test.update({lab : fn})
        
            
        nsamples = np.sum([len(dataset.filenames[lab]) for lab in labels])    
        nsamples_train = np.sum([len(self.filenames_train[lab]) for lab in labels]) 
        nsamples_valid = np.sum([len(self.filenames_valid[lab]) for lab in labels]) 
        nsamples_test = np.sum([len(self.filenames_test[lab]) for lab in labels]) 
        
        print('{} samples from {} classes'.format(nsamples, len(labels)))
        print('{}, {}, {} samples of train, valid, test'.\
            format(nsamples_train, nsamples_valid, nsamples_test))
       
            

if __name__ == '__main__':    
    path_dataset = '/home/joans/Documents/docencia/master/exercise deep metric learning/datasets/lfwcrop_color_20/'
    # path_dataset = '/home/joans/Documents/exercise deep metric learning/datasets/lfwcrop_color_20/'
    np.random.seed(0)
    """ to get the same batches at each execution """
    
    ds = Dataset(path=path_dataset, image_extension='ppm')
    ratios = {'train':0.6, 'valid':0.1, 'test':0.3}
    split = Split(ds, ratios)
    ds_train = Dataset(filenames=split.filenames_train)
    ds_valid = Dataset(filenames=split.filenames_valid)
    ds_test = Dataset(filenames=split.filenames_test)
    
    from dataset_pairs import Dataset_pairs
    
    prob_same_class = 0.5
    batch_size = 15
    ds_pairs_train = Dataset_pairs(ds_train, prob_same_class)
    ds_pairs_valid = Dataset_pairs(ds_valid, prob_same_class)
    ds_pairs_test = Dataset_pairs(ds_test, prob_same_class)
    
    batch = ds_pairs_train.next_batch(batch_size)
    batch.show()
    
    
    