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
from sklearn.decomposition import PCA

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

def Data_reduction(x_train, percentage_threshold):
    '''
    Output:
        x_train_filtered: the resulting training data after reducing parameters
    '''
    return x_train_filtered
    

def Dimensionality_reduction_PCA(x_train, dimensions):
    '''
    This function performs PCA on training data and returns the
    reduced dimensionality array.
    Inputs: 
        x_train: the training data input
        dimensions: the number of wanted dimensions
    Output: 
        x_train_reduced: the array with reduced dimensionsionality to 
        the number given by the dimensions paramter
    '''
    pca = PCA(n_components = dimensions)
    x_train_reduced = pca.fit_transform(x_train) #fit and transform the training input data
    return x_train_reduced

def cross_validating_randomforest(model, x_train, y_train):
    '''Eric, write your code for random forest fitting with k-fold cross-validation here
    I'm thinking this function needs to output some sort of classification error metric
    that we can then use to evaluate the parameters in the function below
    '''
    return test_score

def perform_randomforest_sensitivity_analysis(x_train_reduced, y_train, paramgrid):
    '''
    This function will perform RandomForestClassification with a variation
    of different parameter values. It will output the test scores for each parameter value. 
    Inputs: 
        x_train : training data input
        y_train : training data output
        paramgrid : dictionary of the parameter values that will be evaluated
    Outputs: 
        classification_error : a list of all the calculated classification errors
    '''
    classification_error = [] #define list of classification errors
    for parameter in paramgrid: #go through each parameter key
        classification_error_sublist = []
        for value in paramgrid[parameter]:
            kwargs = {}
            kwargs[parameter]=value #Create a dictionary of the parameter to specific value
            model = RandomForestClassifier(criterion = 'gini')
            model.set_params(**kwargs) #pass dictionary to set parameter of model
            test_score = cross_validating_randomforest(model, x_train, y_train) #Perform the model fit and return the test_scorre
            classification_error_sublist.append(test_score) #Now add test_score to the output list of errors
        classification_error.append(classification_error_sublist)
    return classification_error

#Load data
train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year

#Here create a loop that fits randomforest to multiple different PCA dimensions
#Plot the dimension number versus test score to find sweet-spot number of dimensions


#Here use perform_randomforest_sensitivity_analysis to train on the data with optimal number of dimensions
#Plot the parameter value versus classification error to find parameter sweet-spot
param_grid = { 
            "n_estimators" : list(np.arange(1000,10000,1000)),
            "max_features" : ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
            "max_depth" : list(np.arange(2,20)), #This is what we used in the problem set
            "min_leaf_node" : list(np.arange(1,26)) #This is what we used in the problem set
            }

#Now perform actual model fit with optimized parameters and dimensions

