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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score

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
    # fairly slow implementations with for loops. May try to use np to speed up.
    shape = x_train.shape

    for i in range(shape[1]):
        col = x_train[:,i]
        unique, counts = np.unique(x, return_counts=True)
        # combine classes and counts. Maybe use for display purposes later?
        # I'm using ## as comment for code
        ## frequencies = np.asarray((unique, counts)) 

        maxPercent = np.max(counts) / shape[1]

        # if the percentage of a certain class is high enough, then 
        # slice. Haven't written slice yet. 
        # FIX: write slice part. 
        if(maxPercent > percentage_threshold):
            pass

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

# decently useful makeplot function. Not very customizable. 
def makePlot(x, y, x_label, y_label, gentitle):
    plt.figure()
    plt.plot(x, y, color = 'c', linewidth = 1, label = y_label)
    plt.legend(loc = 'best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(gentitle)

    plt.show()

def cross_validating_randomforest(model, x_train, y_train):
    '''Eric, write your code for random forest fitting with k-fold cross-validation here
    I'm thinking this function needs to output some sort of classification error metric
    that we can then use to evaluate the parameters in the function below. 
    '''
    '''
    Wow this is a lot thanks Eva. 
    This function does cross validation for an sklearn model. Not usable for Keras though. 
    Output: tuple w/ [Classification Error, AUC loss]
    '''

    # basic cross val scores using cross validation
    # should return array of classification accuracy
    cv_accuracy = cross_val_score(model, x_train, y_train, cv=5)

    # Get probability scores
    pred_prob = model.predict_proba(x_train)[:,1]
 
    # plot ROC curve
    roc_curve_ = roc_curve(y_train, pred_prob)

    # Get the area under the ROC curve for general score. 
    roc_auc_score_ = roc_auc_score(y_train, pred_prob)

    # plot the roc curve
    makePlot(roc_curve_[0], roc_curve_[1], 'FPR', 'TPR', 'ROC Curve')

    # then get area under the ROC curve for measure of how good separation

    return (cv_accuracy, roc_auc_score_)

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
model = RandomForestClassifier(criterion = 'gini')
model.fit(x_train_reduced, y_train)
(cv_accuracy, roc_auc_score_) = cross_validating_randomforest(model, x_train_reduced, y_train)

#dimensions_lst = [10,100,379]
#for dimensions in dimensions_lst:
#    x_train_reduced = Dimensionality_reduction_PCA(x_train, dimensions)
#    model = RandomForestClassifier(criterion = 'gini')
#    model.fit(x_train_reduced, y_train)
#    (cv_accuracy, roc_auc_score_) = cross_validating_randomforest(model, x_train_reduced, y_train)
    

    


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

