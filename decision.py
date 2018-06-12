# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:22:14 2018

@author: Shubham
"""

# One-Class SVM for anomaly detection
import numpy as np
import sys
import os
import numpy.linalg as LA
from sklearn import utils, svm, metrics
import matplotlib.pyplot as plt

# custom functions
from data_load import *
from model import *
from squeezenet import *
from AE import *

# One-Class SVM class functions
class Decision():
    
    # One Class SVM parameter intialization
    # labels_test for accuracy
    def __init__(self, nu, kernel, gamma, base_vec, labels_train, labels_test):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.base_vec = base_vec
        self.labels_train = labels_train
        self.labels_test = labels_test
    
    def MSE(self):
        X_train = []
        for batch in batch_size:
            
    
    # train the One-Class SVM for a specific signature
    def fit_model(self):
        clf = svm.OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        X_train = self.MSE()
        Y_train = self.labels
        clf.fit(X_train, Y_train)
        
    def predict(self):
        
        