"""
Created on Thu Jun  8 22:50:06 2017

@author: joans

Siamese network based on this implementation  
    https://github.com/ywpkwon/siamese_tf_mnist
by Youngwook Paul Kwon, tested on MNIST. I've replaced his branches (a pair of 
MLPs) by a modified VGG16 and added dropout.
"""
import os
import time
import numpy as np

import tensorflow as tf 

from vgg16 import Vgg16



class Siamese():
    def __init__(self, dim_embedding, image_size, margin):
        self.dim_embedding = dim_embedding
        self.image_size = image_size
        self.margin = margin
        datestr = time.asctime().replace(' ','_').replace(':','_')        
        self.path_experiment = 'experiments/'+datestr
               
        tf.reset_default_graph()
        self.make_placeholders()
        self.make_model()
        self.make_loss()
                   
        

    def make_placeholders(self):
        shape_x = [None, self.image_size, self.image_size, 3]
        self.x1 = tf.placeholder(tf.float32, shape_x)
        self.x2 = tf.placeholder(tf.float32, shape_x)
        self.y = tf.placeholder(tf.float32, [None])
        self.keep_prob_fc = tf.placeholder(tf.float32)


    def make_model(self):
        with tf.variable_scope("siamese") as scope:
            self.out1 = self.network_branch(self.x1)
            scope.reuse_variables()
            self.out2 = self.network_branch(self.x2)



    def network_branch(self, x):
        self.vgg16 = Vgg16(x, self.keep_prob_fc, width_fc=512, 
                           num_features=self.dim_embedding)
        return self.vgg16.out
                
                

    def make_loss(self):
        self.loss = self.loss_with_spring(self.out1, self.out2)




    """ contrastive loss, original or modified with margin for similar pairs """
    def loss_with_spring(self, out1, out2, two_margins=False):
        eucd2 = tf.pow(out1 - out2, 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        """ ||CNN(p1i)-CNN(p2i)||_2^2 """
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        """ ||CNN(p1i)-CNN(p2i)||_2 """
        margin = tf.constant(self.margin, name="margin")
        """ self.y : 0 = same class, 1 = different class """

        if two_margins:
            """ with additional margin for same class, stops pulling
            closer similar samples if their distance <= second margin.
            we set it to 1/20th of the other margin which is 1.0 """
            margin_same = tf.constant(0.05, name="margin_same")
            same = tf.multiply(tf.subtract(1.0, self.y, name="1-yi"),
                               tf.pow(tf.maximum(0.0, eucd - margin_same), 2),
                               name="same_class")
        else:
            """ original version, keeps pulling closer samples from
            same class all the time """
            same = tf.multiply(tf.subtract(1.0, self.y, name="1-yi"), eucd2,
                               name="same_class")

        different = tf.multiply(self.y, tf.pow(tf.maximum(0.0, margin - eucd), 2),
                           name="different_class")
        loss = tf.reduce_mean(same + different, name="loss")
        return loss


    def train(self, dataset_pairs, learning_rate, batch_size, max_steps, 
              keep_prob_fc, show_loss_every_steps, save_weights_every_steps):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for step in range(1,max_steps+1):
                batch = dataset_pairs.next_batch(batch_size)
                _, bloss = sess.run([train_op, self.loss],
                             feed_dict = {
                                 self.x1: batch.x1,
                                 self.x2: batch.x2,
                                 self.y: batch.y.astype('float'),
                                 self.keep_prob_fc: keep_prob_fc,
                             })

                if step % show_loss_every_steps == 0:
                    print ('step {}, loss {}'.format(step, bloss))
 
                if step % save_weights_every_steps == 0:
                    self.save_weights(sess, saver)



    def save_weights(self, sess, saver):
        if not os.path.isdir(self.path_experiment):
            os.makedirs(self.path_experiment)
            print('made output directory {}'.format(self.path_experiment))

        saver.save(sess, os.path.join(self.path_experiment, 'model.ckpt'))
        print('saved weights at {}'.format(self.path_experiment))



    def load_weights(self, path_experiment, saver, sess):
        checkpoint_dir = path_experiment
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('loaded weights from {}'.format(ckpt.model_checkpoint_path))
        else:
            print('ERROR: no checkpoint found for path {}'.format(path_experiment))
            assert False



    def inference_one_image(self, ima, path_experiment):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)

        emb =  self.out1.eval({self.x1: [ima], self.keep_prob_fc: 1.0})
        point = list(emb[0])
        return point
    
    
    def inference_dataset(self, ds, path_experiment): 
        labels = []
        points = []
        images = []
        
        saver = tf.train.Saver()
        i = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)
            
            for (ima, lab) in ds.samples():
                emb = self.out1.eval({self.x1: [ima], self.keep_prob_fc: 1.0})
                points.append(list(emb[0]))
                labels.append(lab)
                images.append(ima)
                
                i += 1
                if i%100==0:
                    print(i)
                                
        return np.array(labels), np.array(points), np.array(images)

