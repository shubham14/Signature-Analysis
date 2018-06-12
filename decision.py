# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:22:14 2018

@author: Shubham
"""

# One-Class SVM for anomaly detection
import numpy as np
import sys
import os
from sklearn import utils, svm, metrics
import matplotlib.pyplot as plt

# custom functions
from data_load import *
from model import *
from squeezenet import *
from AE import *

class Decision():
    
    # One Class SVM parameter intialization
    def __init__(self, nu, kernel, gamma):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
    
    def fit_model(self):
        