# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:33:47 2017

@author: joans
"""

import shutil
import os


"""
This is to save the source code in the same folder where the results are,
so as to be able to reproduce them at any time and also to check with which
parameters we've generated them.
"""
def save_code(path):
    assert os.path.isdir(path)       
    """ copy all the sources """
    os.system('tar cvfz sources.tgz *.py')
    shutil.move('sources.tgz', path)
    print('saved source code to '+path+'/sources.tgz')


"""
How to save figures in 'native format' like .fig in Matlab :
http://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle/
"""
import pickle

def save_figure(fig_handle, fname):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
        
    pickle.dump(fig_handle, open(fname, 'w'))
    
    
def load_figure(fname):
    assert fname.endswith('.pkl')
    fig_handle = pickle.load(open(fname, 'rb'))
    return fig_handle
    # to show it do fig_handle.show() perhaps with block=False as parameter
    
""" 
scikit-image's resize() changes from uint8 min...max of input to float 0...1 
and we don't want that but want to keep the original min...max scale and
the unit8 format """
import numpy as np
from skimage.transform import resize

def resize_uint8(ima, new_shape):
    """ new_shape is [nrows, ncols] """
    assert type(new_shape) in [list, tuple]
    assert len(new_shape) == 2
    return np.round(255*(resize(ima/255.0, new_shape, order=1))).astype(np.uint8)
    


    