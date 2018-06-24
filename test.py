# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:43:56 2018

@author: Shubham
"""

import tensorlfow as tf
import numpy as np
import cv2
from PIL import image
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.contrib.layers as layers
from skimage import transform

dataPath = r'C:\Users\Shubham\Desktop\Dataset_4NSigComp2010\TrainingSet\Genuine'

def AutoEncoder(inputs):
    # encoder architexture
    net = layers.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = layers.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = layers.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    
    # decoder architecture
    net = layers.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = layers.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = layers.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

def train():
    inputs = tf.placeholder(tf.float32, (None, 1670, 2535, 1))
    outputs = AutoEncoder(inputs)
    
    loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    init = tf.global_variables_intializer()
    
batch_per_ep = len(dataSet) // batch_size

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):
        for batch_n in range(batch_per_ep):
                        batch_img, batch_label = tf.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 1670, 2535, 1))               # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))        