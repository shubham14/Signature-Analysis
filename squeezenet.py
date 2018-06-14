# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:13:41 2018

@author: Shubham
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# import tensorflow libraries
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_loss2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope

activations = []
NORM = 100

# class instantiating Squeezenet model 
class SqueezeNet():
    
    def __init__(self, inputs, squeezeTo, expandTo):
        self.inputs = inputs
        self.squeezeTo = squeezeTo
        self.expandTo = expandTo
    
    def fire(self):
        h = squeeze(self.inputs, squeezeTo)
        h = expand(h, expandTo)
        h = tf.clip_by_norm(h, NORM)
        activations.append(h)
        
    # squeeze network
    def squeeze(self):
        with tf.name_scope('squeeze'):
            inputSize = self.inputs.get_shape().as_list()[3]
            w = tf.Variable(tf.truncated_normal([1,1,inputSize,squeezeTo]))
            h = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
        return h
        
    # exapnd network concatenate a 1 X 1 and 3 X 3 network
    def expand(self):
        with tf.name_scope('expand'):
            squeezeTo = self.inputs.get_shape().as_list()[3]
            w = tf.Variable(tf.truncated_normal([1,1,squeezeTo,expandTo]))
            h1x1 = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
            w = tf.Variable(tf.truncated_normal([3, 3, squeezeTo, expandTo]))
            h3x3 = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
            h = tf.concat(3, [h1x1, h3x3])
        return h
    
    filters = [8, 8, 16, 16, 24, 24, 32, 32]
    squeezes= [2, 2, 4, 4, 6, 6, 8, 8]
    with tf.name_scope('conv1'):
        w = tf.Variable(tf.truncated_normal([3,3,1,64]))
        h = tf.nn.relu(tf.nn.conv2d(activations[-1],w,[1,2,2,1],'SAME'))
        activations.append(h)

    with tf.name_scope('maxpool1'):
        h = tf.nn.max_pool(activations[-1],[1,3,3,1],[1,2,2,1],'SAME')
        activations.append(h)
    
    for i in range(0,2):
        with tf.name_scope('fire'+str(i+2)):
            fire(activations[-1],squeezes[i],filters[i])
    
    with tf.name_scope('maxpool2'):
        h = tf.nn.max_pool(activations[-1],[1,3,3,1],[1,2,2,1],'SAME')
        activations.append(h)
    
    for i in range(2,4):
        with tf.name_scope('fire'+str(i+2)):
            fire(activations[-1],squeezes[i],filters[i])
    
    with tf.name_scope('maxpool3'):
        h = tf.nn.max_pool(activations[-1],[1,3,3,1],[1,2,2,1],'SAME')
        activations.append(h)
    
    for i in range(4,7):
        with tf.name_scope('fire'+str(i+2)):
            fire(activations[-1],squeezes[i],filters[i])
    
    with tf.name_scope('dropout'):
        h = tf.nn.dropout(activations[-1],keep_prob)
        activations.append(h)
    
    with tf.name_scope('conv10'):
        input_shape = activations[-1].get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1,1,input_shape,classCount]))
        h = tf.nn.relu(tf.nn.conv2d(activations[-1],w,[1,1,1,1],'SAME'))
        activations.append(h)
    
    with tf.name_scope('avgpool'):
        input_shape = activations[-1].get_shape().as_list()[2]
        h = tf.nn.avg_pool(activations[-1],[1,input_shape,input_shape,1],[1,1,1,1],'VALID')
        h = tf.squeeze(h,[1,2])
        activations.append(h)
    
    y_conv = tf.nn.softmax(activations[-1])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(activations[-1], y)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tf.summary.scalar('accuracy',accuracy)
