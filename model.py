# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:57:18 2018
To add squeezenet architecture for 
@author: Shubham
"""

import numpy as np
import pandas as pd
import sys
import h5py
import os
from data_load import DataPrepper

# import tensorflow libraries
import tensorflow as tf 
import tensorflow.contrib.slim as slim

# CNN model
def Model():
    
    # initializing network parameters
    def __init__(self, lr, epochs, batch_size, is_training, img_size,
                 n_filters, kernel_size, pool_size):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_training = is_training
        self.img_size = img_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        
    # architecture built on tensorflow slim
    # outputs are binary in nature
    # output the output layer as well as the final feature vector
    def build_model(self, x, y):
        batch_norm_params = {'is_training': self.is_training,
                      'decay': 0.9,
                      'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
            x = tf.reshape(x, [-1, self.img_size, self.img_size, -1])
            
            # define the network
            net = slim.conv2d(x, self.n_filters, [self.kernel_size, self.kernel_size],
                              scope='conv1')
            net = slim.max_pool2d(net, [self.pool_size, self.pool_size],
                                  scope='pool1')
            net = slim.conv2d(x, 2 * self.n_filters, [self.kernel_size, self.kernel_size],
                              scope='conv2')
            net = slim.max_pool2d(net, [self.pool_size, self.pool_size],
                                  scope='pool2')
            net = slim.flatten(net, scope='flatten3')
            
            # fully connected layers
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=self.is_training, 
                               scope='dropout')
            output = slim.fully_connected(net, 2, activation_fn=None, 
                                          normalizer_fn=None, scope='output')

            return output, tf.reduce_mean(net)