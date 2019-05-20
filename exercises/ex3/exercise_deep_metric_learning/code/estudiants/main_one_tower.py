#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:17:44 2018

@author: joans
"""
import os
import pickle

import numpy as np

from one_tower import One_tower             
from dataset import Dataset
from split import Split
from plot_embedding import plot_embedding, plot_tsne


def get_path_and_extension(dataset_name):
    if dataset_name=='lfw':
        path = '/home/joans/Documents/exercise_deep_metric_learning/datasets/lfwcrop_color_by_dirs/'
        extension = 'ppm'
    elif dataset_name=='tsinghua':
        path = '/home/joans/Documents/exercise_deep_metric_learning/datasets/tsinghua_resized/'
        extension = 'png'
    else:
        assert False
        
    return path, extension


def nearest_neighbor(points, labels, images, p, lab, ima, show_images=False):
    import matplotlib.pyplot as plt
    idx = np.argmin(np.sum((p-points)**2,axis=1))
    
    if show_images:
        # print('nearest point to {} is {}'.format(p, points[idx]))
        print('{} is {}'.format(lab, labels[idx]))
        plt.figure(lab)
        ax = plt.subplot(1,2,1)
        plt.imshow(ima)
        plt.axis('off')
        ax.set_title(lab)
        ax = plt.subplot(1,2,2)
        plt.imshow(images[idx])
        plt.axis('off')
        ax.set_title(labels[idx])

        plt.show(block=False)
    
    return labels[idx]
    

def compute_centers(labels, points):
    points_centers = []
    labels_centers = np.unique(labels)
    for lab in labels_centers:
        idx = np.where(labels==lab)
        c = np.mean(points[idx], axis=0)
        points_centers.append(c)
        # print([lab, c])
        
    points_centers = np.array(points_centers)
    return labels_centers, points_centers
    

""" 
Data 
"""
min_images = 20
""" 20 images =>  62 identities in lfw,  81 classes in tsinghua
    10 images => 158                  , 103
     5 images => 423 """
dataset_name = 'tsinghua' # 'tsinghua' # 'lfw'
if dataset_name == 'lfw':
    data_augmentation = True # horizontal flip
else:
    data_augmentation = False

path_dataset, image_extension = get_path_and_extension(dataset_name)

ratios = {'train':0.80, 'valid':0.0, 'test':0.20}
assert np.abs(1.0 - np.sum(list(ratios.values()))) < 1e-6


""" 
Network 
"""
dim_embedding = 64
""" with 2 you can use plot_embedding() but retrieval won't be as good as with
8, 16 ... If >2 you need to project to two dimensions using t-SNE of scikit-learn,
or much more nicer but more difficult to set up, using Tensorboard's t-SNE or pca """
image_size = 64
margin = 1.0
type_loss = 'triplet' # 'contrastive','triplet'
if type_loss=='contrastive':
    type_mining = None
    """ P, K contrastive, we'll have aprox (PK)^2/2 pairs, and a ratio
    of 1/P pairs of same class """
    P, K = 4, 25 # 10, 20 # 40, 10

if type_loss=='triplet':
    type_mining = 'batch all'
    """ 'batch all', 'batch all nonzero mean' , 'batch hard loss' """
    """ P, K triplet, we'll have aprox P^2 K^3  triplets """
    P, K = 10, 6 # 5, 3



""" 
Learning 
"""
learning_rate = 1e-5
max_steps = 500*1000
keep_prob_fc = 1.
show_loss_every_steps = 10
save_weights_every_steps = int(max_steps/10)
""" this is to be able to see preliminary results while training """


""" 
Hardware 
"""
gpu = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
""" otherwise tensorflow allocates all the gpus """


""" 
What to do 
"""
make_new_split = False
seed = 0
""" set a different seed each time to get a new split """
train = True
compute_and_plot_embedding = False
if compute_and_plot_embedding:
    """ min_images 20, contrastive, P=5, K=3, dim=64, 100 K steps
    -> nn 83  %, ncm 83 % """
    #dim_embedding = 64
    #path_experiment = 'experiments/Thu_Mar__8_23_51_33_2018'

    """ min_images 10, dim=64, triplet L_{BA}, P=5, K=3, 500 K steps
    -> nn 74 %, ncm 74 % """
    #dim_embedding = 64
    #path_experiment = 'experiments/Sat_Mar_10_12_10_01_2018'

    """ idem, L_{BA \neq 0}, 200 K -> nn 62 %, ncm 62 % """
    #dim_embedding = 64
    #path_experiment = 'experiments/Fri_Mar__9_15_39_11_2018'

    """ L_{Batch hard loss} 500 K, P=5, K=3 => 15 termes nomes
    -> a 100K nn 69 %, ncm 69 %, a 500 K 78 %, 78 %"""
    #dim_embedding = 64
    #path_experiment = 'experiments/Sun_Mar_11_15_15_48_2018'
    
    """ L_{Batch hard loss}, 500 K, P=10, K=6,
    -> a 300K nn 78 %, ncm 78 %, a 500 K 79.95 %, 80.26 % """
    #dim_embedding = 64
    #path_experiment = 'experiments/Sun_Mar_11_22_37_47_2018'

    """ min_images 5, P=10, K=6, L_{Batch hard loss} 500 K
    -> a 300K nn 65 %, ncm 65 %, a 500 K 67.21 %, 67.66 %"""
    dim_embedding = 64
    path_experiment = 'experiments/Sun_Mar_11_20_54_43_2018'

    dataset_name = 'lfw'
    pass


batch_size = P*K

split_name = 'split_'+dataset_name+'_'+str(min_images)+'_'+str(seed)+'.pkl'
if make_new_split:
    print('making new split {}'.format(split_name))
    np.random.seed(seed)
    ds = Dataset(path=path_dataset, image_extension=image_extension, 
                 min_images=min_images)
    split = Split(ds, ratios)
    with open(split_name,'wb') as f:
        pickle.dump(split, f)

    print('saved split for {} and min_images={}, set make_new_split to False now'.\
        format(dataset_name, min_images))

else:
    print('loading existing split {}'.format(split_name))
    with open(split_name,'rb') as f:
        split = pickle.load(f)
        

ds_train = Dataset(filenames=split.filenames_train,
                   data_augmentation=data_augmentation)
ds_test = Dataset(filenames=split.filenames_test)

 
if train:
    one_tower = One_tower(dim_embedding, image_size)
    one_tower.train(margin, type_loss, type_mining, ds_train, learning_rate, max_steps,
          P, K, keep_prob_fc, show_loss_every_steps, save_weights_every_steps)
    print('path experiment {}'.format(one_tower.path_experiment))     
   
if compute_and_plot_embedding:
    """ compute and save embeddings of the datasets.
        plots can not be done inside a screen """
    one_tower = One_tower(dim_embedding, image_size) 
    """ batch_size , type_loss do not matter for inference """
    labels_train, points_train, images_train = one_tower.inference_dataset(ds_train, path_experiment)
    labels_test, points_test, images_test = one_tower.inference_dataset(ds_test, path_experiment)

    if False:
        if dim_embedding==2:
            plot_embedding(labels_train, points_train, images_train, interactive=False)     
            plot_embedding(labels_test, points_test, images_test, interactive=False)    
        else:
            plot_tsne(np.array(list(labels_train)+list(labels_test)), 
                      np.vstack([points_train, points_test]), lr=100)
            """ this take some minutes for tsinghua
            Read https://distill.pub/2016/misread-tsne/ for how to use t-sne !!
            In all my tests for dim=16 I've just got a ball, no clusters. Better
            change to dim=8 ?
            """

    """ assess accuracy on train set, with nearest-neighbor is obviously 100% always """
    labels_pred_nn_train = []
    labels_pred_ncm_train = []
    labels_centers, points_centers = compute_centers(labels_train, points_train)
    for p, lab, ima in zip(points_train, labels_train, images_train):
        pred_nn = nearest_neighbor(points_train, labels_train, images_train, p, lab, ima, show_images=False)
        labels_pred_nn_train.append(pred_nn)
        pred_ncm = nearest_neighbor(points_centers, labels_centers, None, p, lab, ima, show_images=False)
        labels_pred_ncm_train.append(pred_ncm)
        
    acc_nn = np.sum(labels_pred_nn_train==labels_train)/float(len(labels_train))
    print('accuracy train nn {} %'.format(100*acc_nn))
    acc_ncm = np.sum(labels_pred_ncm_train==labels_train)/float(len(labels_train))
    print('accuracy train ncm {} %'.format(100*acc_ncm))

    """ assess accuracy on test set """
    labels_pred_nn_test = []
    labels_pred_ncm_test = []
    for p, lab, ima in zip(points_test, labels_test, images_test):
        pred_nn = nearest_neighbor(points_train, labels_train, images_train, p, lab, ima, show_images=False)
        labels_pred_nn_test.append(pred_nn)
        pred_ncm = nearest_neighbor(points_centers, labels_centers, None, p, lab, ima, show_images=False)
        labels_pred_ncm_test.append(pred_ncm)

    acc_nn = np.sum(labels_pred_nn_test==labels_test)/float(len(labels_test))
    print('accuracy test {} %'.format(100*acc_nn))
    acc_ncm = np.sum(labels_pred_ncm_test==labels_test)/float(len(labels_test))
    print('accuracy test ncm {} %'.format(100*acc_ncm))
    
    if dataset_name == 'lfw':
        """ Finally, look for the look-alikes of each of my images """
        my_ds = Dataset(path='/home/joans/Documents/exercise_deep_metric_learning/my_images',
                        image_extension='ppm', min_images=1, data_augmentation=False)
        labels_my, points_my, images_my = one_tower.inference_dataset(my_ds, path_experiment)

        if dim_embedding==2:
            plot_embedding(labels_my, points_my, images_my, interactive=False)

        for p, lab, ima in zip(points_my, labels_my, images_my):
            nearest_neighbor(points_train, labels_train, images_train, p, lab, ima, show_images=True)
    

