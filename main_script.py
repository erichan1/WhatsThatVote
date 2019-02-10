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
    x_train_filtered = np.delete(x_train, delete_cols, 1)
    return x_train_filtered

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

    # roc auc score using 5fold cross val
    roc_auc_scores = cross_val_score(model, x_train, y_train, cv=5, scoring = 'roc_auc')

    # Get probability scores
    ## pred_prob = model.predict_proba(x_train)[:,1]

    # plot ROC curve
    ## roc_curve_ = roc_curve(y_train, pred_prob)

    # plot the roc curve
    ## makePlot(roc_curve_[0], roc_curve_[1], 'FPR', 'TPR', 'ROC Curve')

    # then get area under the ROC curve for measure of how good separation

    return (cv_accuracy, roc_auc_scores)

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
        n=1
        for value in paramgrid[parameter]:
            print('update {}'.format(n))
            n+=1
            kwargs = {}
            kwargs[parameter]=value #Create a dictionary of the parameter to specific value
            model = RandomForestClassifier(criterion = 'gini')
            model.set_params(**kwargs) #pass dictionary to set parameter of model
            cv_accuracy, auc_accuracy = cross_validating_randomforest(model, x_train, y_train) #Perform the model fit and return the test_scorre
            classification_error_sublist.append(np.mean(auc_accuracy)) #Now add test_score to the output list of errors
        classification_error.append(classification_error_sublist)
    
    return classification_error

#Load data
train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year
test_data_2012 = load_data('test_2012.csv')
x_test_2012 = train_data[:,3:]

x_train_reduced = data_reduction(x_train, 0.98)
print(x_train.shape)
print(x_train_reduced.shape)

x_test_reduced = data_reduction(x_test, 0.98)
print(x_test.shape)
print(x_test_reduced.shape)

#Here use perform_randomforest_sensitivity_analysis to train on the data with optimal number of dimensions
#Plot the parameter value versus classification error to find parameter sweet-spot
#paramgrid = { 
         #   "max_features" : ["auto", "sqrt", "log2"],
          #  "min_samples_leaf" : list(np.arange(1,100,5)) #This is what we used in the problem set
         #   }
          #  "n_estimators" : list(np.arange(10,200,50)),
#            "max_features" : ["auto", "sqrt", "log2"],
#            "bootstrap": [True, False],
#            "max_depth" : list(np.arange(2,20)), #This is what we used in the problem set
#            "min_leaf_node" : list(np.arange(1,26)) #This is what we used in the problem set
#            }

#classification_error_n_estimators = perform_randomforest_sensitivity_analysis(x_train_reduced, y_train, paramgrid)

#Now perform actual model fit with optimized parameters and dimensions
model = RandomForestClassifier(criterion = 'gini')
model.set_params(n_estimators=500,max_features='auto',min_samples_leaf=25)
#model.fit(x_train_reduced,y_train)
(cv_accuracy,roc)=cross_validating_randomforest(model, x_train_reduced, y_train)
