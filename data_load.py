# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:54:17 2018

@author: Shubham
"""

import numpy as np
import os
from glob import glob
import sys
import cv2 # for image resize
import tensorflow as tf

# class containing the data loader for the tensorflow deep architecture
class DataPrepper():
    
    def __init__(self, filename, img_size):
        self.filename = filename
        self.img_size = img_size
        
    # select .jpg and .png fiels from a file path
    def data_load(self):
        data_types = ["*.png", "*.jpg"]
        d_image = []
        for d_type in data_types:
            d_image.extend(glob(d_type))
        return d_image
    
    # to check if this would work for large amounts of images
    # crop out the image and convert it to black and white
    def data_process(self):
        img_list = self.data_load()
        img_res = []
        for img in img_list:
            img = cv2.imread(img)
            res = cv2.resize(img, dsize=(self.img_size, self.img_size), 
                             interpolation=cv2.INTER_CUBIC)
            gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            img_res.append(res)
        return img_res