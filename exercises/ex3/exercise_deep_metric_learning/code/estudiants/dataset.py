# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:38:08 2017

@author: joans
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from skimage.io import imread


class Dataset():
    """ this is to actually have two constructors """
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        named_params = list(kwargs.keys())
        if 'path' in named_params and \
           'image_extension' in named_params and \
           'min_images' in named_params:
                path = kwargs['path']
                image_extension = kwargs['image_extension']
                min_images = kwargs['min_images']
                self.__init_from_path(path, image_extension, min_images)

        elif 'filenames' in named_params :   
            filenames = kwargs['filenames']
            self.__init_from_filenames(filenames)
        else:
            assert False

        if 'data_augmentation' in named_params :
            self.data_augmentation = kwargs['data_augmentation']
        else:
            self.data_augmentation = False

        self.unique_labels = np.sort(np.array(list(self.filenames.keys())))
        self.num_labels = len(self.unique_labels)

        self.epochs_completed = 0
        self.index_in_epoch = 0


    """ the normal constructor that makes a dataset with all the classes in
    a certain folder, one subfolder per class. min_images is to select only
    those classes with that number of images at least """
    def __init_from_path(self, path, image_extension='ppm', min_images=1):
        self.path = path
        self.image_extension = image_extension    
        self.min_images = min_images
        self.filenames = self.__make_filenames()


    """ to make a dataset from a selection of labels and/or samples for each
    label, coming from another dataset built the normal way. this is employed 
    to make a train/test/split, see Split.py . In this way we do not need to
    makr a hard partition of train/val/test samples of each class by moving
    them to different folders and can change the ratios and randomize the
    parition """
    def __init_from_filenames(self, filenames):
        first_key = list(filenames.keys())[0]
        fname = filenames[first_key][0]
        self.image_extension = fname.split('.')[-1]     
        self.filenames = filenames


    def __make_filenames(self):        
        filenames = {}
        """ a dictionary of (label, list of full path filenames) """
        dir1 = os.listdir(self.path)
        dir1.sort() 
        """ labels will be sorted by name """
        for fname in dir1:
            if os.path.isdir(os.path.join(self.path, fname)):
                lab = fname
                fnames = glob.glob(os.path.join(self.path,lab)+'/*.' \
                    +self.image_extension)
                num_images = len(fnames)
                assert num_images>0
                if num_images>=self.min_images:
                    filenames.update({lab : fnames})
                    
        return filenames


    """ pre-process original image for various reasons, for the moment only to
    normalize it. The output is float in the range [0.0, 1.0]. This is the
    place to do data augmentation, at the moment only horizontal flip.
    """
    def __get_image(self, fname):
        ima = imread(fname)
        assert ima.ndim==3
        assert ima.dtype=='uint8'
        """ it's a  0...255 color image """
        new_ima = ima.astype(np.float)/255.0   
        if self.data_augmentation:
            if np.random.rand()>0.5:
                new_ima = np.fliplr(new_ima)
                               
        return new_ima
            
        
    """ returns the image of a randomly chosen sample from class 'label' """
    def get_random_sample_of_a_class(self, label):
        num_samples_class = len(self.filenames[label])
        idx = np.random.randint(num_samples_class)
        fname = self.filenames[label][idx]         
        return self.__get_image(fname)


    def __filenames_to_arrays(self):
        samples = []
        labels = []
        for lab in self.filenames.keys():
           s = self.filenames[lab]
           samples.extend(s)
           labels.extend(len(s)*[lab])
           
        perm = np.random.permutation(len(samples))
        samples = np.array(samples)[perm]
        labels = np.array(labels)[perm]
        return samples, labels


    """ this is to get the image of all samples in succession, exactly once 
    per sample, for instance to get all samples of a test set partition """
    def samples(self):
        all_samples, all_labels = self.__filenames_to_arrays()
        num_samples = len(all_samples)
        for i in range(num_samples):
            ima = self.__get_image(all_samples[i])
            yield ima, all_labels[i]
    

    """ A batch is built by random sampling with replacement a certain
    class label, and then randomly draw one sample from that class, again with
    replacement. This is different from the standard practice of successively 
    drawing samples so as to cover the whole dataset in one epoch, one sample 
    being drawn just once per epoch. That is, we do not support the concept of 
    epoch. In this way we will get approximately the same number of samples 
    per class, a way to deal with unbalanced datasets """
    def next_batch_uniform(self, batch_size):
        x = []
        y = []
        idx = np.random.randint(low=0, high=self.num_labels, size=batch_size)
        labels = np.array(self.unique_labels)[idx]
        for lab in labels:
             x.append(self.get_random_sample_of_a_class(lab))
             y.append(list(self.unique_labels).index(lab))

        return Batch(np.array(x), np.array(y), np.array(self.unique_labels))


    """ Returns a batch of pk images, from p randomly selected classes *with
    replacement* and k images per class """
    def next_batch_pk(self, P, K):
        x = []
        y = []
        """ this is for random sampling classes *with replacement*
          idx = np.random.randint(low=0, high=self.num_labels, size=self.P)
        but the batch hard mining technique relies on getting K images of
        P *different* classes at each batch => without replacement """
        assert P <= self.num_labels
        idx = np.random.permutation(self.num_labels)[:P]
        
        labels = np.array(self.unique_labels)[idx]
        for lab in labels:
            for i in range(K):
                 x.append(self.get_random_sample_of_a_class(lab))
                 y.append(list(self.unique_labels).index(lab))

        return Batch(np.array(x), np.array(y), np.array(self.unique_labels))
        

    
    """ 
    Below, there is a number of plot functions formerly in Dataset_reader
    """

    def show_examples(self, num_examples_per_class=5, 
                      num_classes_per_figure=6):
        """ show 5 examples for each label, sorted alphabetically """
        labels_by_name = np.sort(self.unique_labels)
        num_classes = self.num_labels
        num_figures = int(np.ceil(num_classes/float(num_classes_per_figure)))
        
        for nf in range(num_figures):
            plt.figure(figsize=(10,20))
            plt.subplot(num_classes_per_figure, num_examples_per_class, 1)
            labels_this_figure = labels_by_name[
                nf*num_classes_per_figure : \
                min((nf+1)*num_classes_per_figure, num_classes) ]
                
            for n,lab in enumerate(labels_this_figure):
                num_samples = len(self.filenames[lab])
                for i in range( min(num_examples_per_class, num_samples) ) :
                    ima = self.get_random_sample_of_a_class(lab)
                    plt.subplot(num_classes_per_figure, num_examples_per_class, 
                                n*num_examples_per_class+i+1)
                    plt.imshow(ima, interpolation='nearest')
                    plt.axis('off')
                    if i==0:
                        plt.title(str(lab))
                        
            plt.show(block=False)


    def show_all_images_one_class(self, label, rows=4, cols=4):   
        fnames = self.filenames[label]
        num_images = len(fnames)
        print('{} images found of class {}'.format(num_images, label))
        num_figures = int(np.ceil(float(num_images)/(rows*cols)))
        print('{} figures'.format(num_figures))
            
        for nf in range(num_figures):
            plt.figure()
            plt.subplot(rows, cols, 1)
            n0 = nf*rows*cols
            n1 = min(num_images, (nf+1)*rows*cols)
            for n,p in zip(range(n0,n1), range(rows*cols)):
                ima = imread(fnames[n])
                plt.subplot(rows, cols, p+1)
                plt.imshow(ima, interpolation='nearest')
                plt.axis('off')
                sample_name = fnames[n].split('/')[-1].split('.')[0]
                plt.title(sample_name, fontsize=8)
                
            plt.show(block=False)
        
        
    def show_one_image_per_class(self,  rows=4, cols=4, save=False):
        images = []
        num_images_per_class = []
        labels = np.sort(list(self.filenames.keys()))
        for lab in labels:
            num_images = len(self.filenames[lab])
            num_images_per_class.append(num_images)
            """ select one of the images randomly, because the can be
            sorted by size """
            n = np.random.randint(0,num_images)
            ima = imread(self.filenames[lab][n])
            images.append(ima)
            print(lab, num_images)
                
        num_figures = int(np.ceil(float(self.num_labels)/(rows*cols)))
        print('{} figures'.format(num_figures))
            
        for nf in range(num_figures):
            fig = plt.figure()
            plt.subplot(rows, cols, 1)
            n0 = nf*rows*cols
            n1 = min(self.num_labels, (nf+1)*rows*cols)
            for n,p in zip(range(n0,n1), range(rows*cols)):
                plt.subplot(rows, cols, p+1)
                plt.imshow(images[n], interpolation='nearest')
                plt.axis('off')
                plt.title(str(labels[n])+' - '\
                          + str(num_images_per_class[n]), fontsize=8)
                
            plt.show(block=False)
            if save:
                fig.savefig('figure_'+str(nf)+'.png',bbox_inches='tight')
        

    def sort_by_num_images(self):        
        all_labels = np.array(list(self.filenames.keys()))
        num_images = np.array([len(self.filenames[lab]) \
            for lab in self.filenames.keys()])
        idx = np.argsort(num_images)[::-1]
        num_images = num_images[idx]
        all_labels = all_labels[idx]
        return all_labels, num_images


    """ graphic showing num images (not always images = one same traffic
    sign exemplar) per class, in decreasing order """        
    def plot_num_images_per_class(self, save=False, suffix_title=''):
        all_labels, num_images = self.sort_by_num_images()
        print('number of images \n{}'.format(num_images))
        num_classes = len(num_images)
        print('total number of classes {}'.format(num_classes))

        fig, ax = plt.subplots()
        x_pos = np.arange(len(all_labels))
        ax.bar(x_pos, num_images, align='center',
                color='green', ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_labels, rotation='vertical', fontsize=8)
        ax.set_xlabel('classes')
        ax.set_ylabel('images')
        ax.set_title('Number of images per class '+suffix_title)
        plt.show(block=False)        
        
        if save:
            fig.savefig('images.png', bbox_inches='tight')
            
            
        
""" 
This is what next_batch() of a Dataset object returns, just two lists, one of
samples and the other of corresponding class labels.
"""
class Batch():
    def __init__(self, x, y, unique_labels=None):
        self.x = x
        self.y = y
        """ this is just to plot or for later reference without a Dataset object """
        self.unique_labels = unique_labels
        
        
    def show(self):
        batch_size = len(self.x)
        plt.figure()
        plt.subplot(1,batch_size,1)
        for i in range(batch_size):
            plt.subplot(1,batch_size,i+1)
            plt.imshow(self.x[i], interpolation='nearest')
            if self.unique_labels is None:
                title = str(self.y[i])
            else:
                title = str(self.unique_labels[self.y[i]])
                
            plt.title(title,fontsize=8)
            plt.axis('off')
            
        plt.show(block=False)
        


def test_plots(ds):
    ds.plot_num_images_per_class()
    ds.show_one_image_per_class()
    label = ds.unique_labels[0]
    print(label)
    ds.show_all_images_one_class(label) 
    ds.show_examples(num_examples_per_class=5)


def test_batches(ds):     
    np.random.seed(seed=0)
    batch_unif = ds.next_batch_uniform(15)
    batch_unif.show()
        
    
def test_samples(ds):
    i = 0
    for ima, label in ds.samples():
        print(label)
        i += 1
        if i>10: 
            break
        

def test_constructor(dataset):
    print('\ntest_constructor()')
    print('three samples of five classes')
    selected_labels = list(dataset.filenames.keys())[:5]
    print('selected labels {}'.format(selected_labels))
    filenames = {lab : dataset.filenames[lab][:3] \
                 for lab in selected_labels}
    ds2 = Dataset(filenames=filenames)
    test_plots(ds2)
    """ caution, this displays a lot of figures """
    test_batches(ds2)
    test_samples(ds2)
    

if __name__ == '__main__':
    dataset_name = 'lfw' # 'lfw' # 'tsinghua'
    if dataset_name == 'lfw':
        path_dataset = '/home/joans/Documents/exercise_deep metric_learning/datasets/lfwcrop_color_by_dirs/'
        image_extension = 'ppm'
    elif dataset_name == 'tsinghua':
        path_dataset = '/home/joans/Documents/exercise_deep_metric_learning/datasets/tsinghua_resized/'
        image_extension = 'png'
    else:
        assert False
    
    np.random.seed(0)
    """ to get the same batches at each execution """
    
    ds = Dataset(path=path_dataset, image_extension=image_extension, min_images=20)
#    print('{} labels in dataset'.format(len(ds.unique_labels)))
#    test_plots(ds)
#    """ caution, this displays a lot of figures """
#    test_batches(ds)
#    test_samples(ds)
    
#    test_constructor(ds)
           
