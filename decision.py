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
    
    def __init__(self):
        pass
    
    def fit_model(self):
        