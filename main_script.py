#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:24:51 2019

CS155 Project 1: Predict voter turnout

@author: Eva Scheller and Eric Han
"""

#Import modules
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

#Define functions
def load_data(filename, skiprows = 1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.
    
    Inputs:
        filename: given as a string.
        
    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=skiprows, delimiter=',')

def PCA(x_train, dimensions):
    pass
    
#Load data
train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year


