#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:07:59 2019

@author: evascheller
"""

#Import modules
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
import csv

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

def data_reduction(x_train, percentage_threshold):
    '''
    Output:
        x_train_filtered: the resulting training data after reducing parameters
    '''
    # fairly slow implementations with for loops. May try to use np to speed up.
    shape = x_train.shape

    # list to hold columns to delete
    delete_cols = []

    for i in range(shape[1]):
        col = x_train[:,i]
        unique, counts = np.unique(col, return_counts=True)
        # combine classes and counts. Maybe use for display purposes later?
        # I'm using ## as comment for code
        ## frequencies = np.asarray((unique, counts)) 

        maxPercent = np.max(counts) / shape[0]

        # if the percentage of a certain class is high enough, then 
        # slice. 
        if(maxPercent > percentage_threshold):
            delete_cols.append(i)
    return delete_cols

def delete_cols(dataset, delete_cols):
    return np.delete(dataset, delete_cols, 1)

def normalize_data(x_data):
    new_x = x_data.copy()
    shape = new_x.shape
    for i in range(shape[1]):
        col = new_x[:,i]
        maxVal = np.max(col)
        new_x[:,i] /= maxVal
    return new_x

train_data = load_data('train_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year

ID_2008 = test_data[:,0]
ID_2012 = test_data_2012[:,0]
cols_delete = data_reduction(x_train, 0.98)
x_train_reduced = delete_cols(x_train, cols_delete)
print(x_train.shape)
print(x_train_reduced.shape)
x_train_normalized = normalize_data(x_train_reduced)






