
"""
Created on Thu Feb  7 17:24:51 2019

CS155 Project 1: Predict voter turnout
This script is used to perform sensitivity analyses of Random Forest

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
    This function takes the input data and returns the columns that need to be deleted
    if one value takes up more than percentage_threshol % of the columns inputs. 
    Essentially, if all values in the column are the same. 
    Input:
        x_train: input data
        percentage_threshold: threshold for discarding data if one value dominates the input of a column
    Output:
        delete_cols: columns that need to be deleted based on threshold
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
    '''
    This function deletes all the columns identified through the data_reduction function. 
    Input:
        dataset: the input data
        delete_cols: the column index for columns that need to be deleted
    Output: 
        the reduced input dataset
    '''
    return np.delete(dataset, delete_cols, 1)

def normalize_data(x_data):
    '''
    This function performs column-wise normalization on the input data. 
    Input: 
        x_data: the reduced input data
    Output: 
        new_x: the normalized input data
    '''
    new_x = x_data.copy()
    shape = new_x.shape
    for i in range(shape[1]):
        col = new_x[:,i]
        maxVal = np.max(col)
        new_x[:,i] /= maxVal
    return new_x

# decently useful makeplot function. Not very customizable. 
def makePlot(x, y, x_label, y_label, gentitle):
    '''
    This function makes a plot of any x-array and y-array pairing. 
    '''
    plt.figure()
    plt.plot(x, y, color = 'c', linewidth = 1, label = y_label)
    plt.legend(loc = 'best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(gentitle)

    plt.show()

def cross_validating_randomforest(model, x_train, y_train):
    '''
    This function performs 5-fold cross validation and returns the cvv accuracy and roc-auc scores.
    It uses the sklearn cross_val_score function
    Input: 
        model: Random Forest model object
        x_train: reduced and normalized training input
        y_train: training data label
    output: 
        cv_accuracy: calculated cv accuracy
        roc_auc_scores: roc-auc scores
        
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
x_test_2012 = test_data_2012[:,3:]
#Load ID columns for 2008 and 2012 test data
ID_2008 = test_data[:,0]
ID_2012 = test_data_2012[:,0]

#Perform data reduction
cols_delete = data_reduction(x_train, 0.98) #Columns to be deleted
x_train_reduced = delete_cols(x_train, cols_delete) #delete columns for training data
print(x_train.shape)
print(x_train_reduced.shape)
x_train_normalized = normalize_data(x_train_reduced) #normalize training data

#Here use perform_randomforest_sensitivity_analysis to train on the data with optimal number of dimensions
#Plot the parameter value versus classification error to find parameter sweet-spot
paramgrid = { 
            "max_features" : ["auto", "sqrt", "log2"],
            "min_samples_leaf" : list(np.arange(1,100,5)),
            "n_estimators" : list(np.arange(10,200,50)),
            "bootstrap": [True, False],
            "max_depth" : list(np.arange(2,20)),
            }

classification_error_n_estimators = perform_randomforest_sensitivity_analysis(x_train_reduced, y_train, paramgrid)

