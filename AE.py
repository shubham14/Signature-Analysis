# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:26:12 2018
Python implementation of AutoEncoders in tensorflow
@author: Shubham
"""
# AutoEncoders applied on R*1 vector, Dense networks used

import numpy as np
import os
import sys

# import tensorflow libraries
import tensorflow as tf
import tensorflow.contrib.layers as layers

# Class containing AutoEncoder architecture
class AE():
    
    # parameter intialization
    def __init__(self, lr=0.001, epochs=50, batch_size=50,
                 is_training=True, num_nodes=1024, **kwargs):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_nodes = num_nodes
        
    # Dense encoder architecture
    def buildEncoder(self, x):
        net = layers.fully_connected(inputs=x, self.num_nodes,
                                     activation=tf.nn.relu, scope='Dense_Enc1')
        net = layers.fully_connected(inputs=net, self.num_nodes/2,
                                     activation=tf.nn.relu, scope='Dense_Enc2')
        net = layers.fully_connected(inputs=net, self.num_nodes/4,
                                     activation=tf.nn.relu, scope='Dense_Enc3')
        return net

    # Dense Decoder architecture
    def buildEncoder(self, enc_x):
        net = layers.fully_connected(inputs=enc_x, self.num_nodes/4,
                                     activation=tf.nn.relu, scope='Dense_Dec1')
        net = layers.fully_connected(inputs=net, self.num_nodes/2,
                                     activation=tf.nn.relu, scope='Dense_Dec2')
        net = layers.fully_connected(inputs=net, self.num_nodes,
                                     activation=tf.nn.relu, scope='Dense_Dec3')
        return net
    
    def train(self):
        sess = tf.Session()
        
        
    
    
        
        
        