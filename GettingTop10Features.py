"""
Created on Thu Feb  7 17:24:51 2019

CS155 Project 1: Predict voter turnout
This script is used to do the final model fit for Random Forest

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


def get_accuracy_differences(x_train_normalized, y_train, x_test_normalized):
    '''
    This function goes through all columns in the input data and calculates the 
    difference in AUC score between performin Random Forest on the full data set 
    and performing Random Forest on data set without one column. It also
    calculates the average difference in predicted probability of the positive 
    class between performin Random Forest on the full data set 
    and performing Random Forest on data set without one column.
    Input:
        x_train_normalized: normalized and reduced input data
        y_train: class labels
        x_test_normalized: normalized and reduced test data
    Output: 
        score_difference: absolute difference in AUC score
        probability_differece: average difference in probability prediction
    '''
    #perform initial model fit on the full data set
    model = RandomForestClassifier(criterion = 'gini')
    (cv_accuracy,roc)=cross_validating_randomforest(model, x_train_normalized, y_train)
    baseline_roc = np.mean(roc)
    model.fit(x_train_normalized, y_train)
    baseline_predict = model.predict_proba(x_test_normalized)[:,1]
    
    score_difference = []
    probability_difference= []
    for i in range(len(x_train_normalized[0])):
        x_train_MissingColumn = delete_cols(x_train_normalized, i) #reduce training data to remove one column at a time
        x_test_MissingColumn = delete_cols(x_test_normalized, i) #reduce test data to remove one column at a time
        
        model = RandomForestClassifier(criterion = 'gini')
        (cv_accuracy,roc)=cross_validating_randomforest(model, x_train_MissingColumn, y_train) #calculate cv accuracy AUC score 
        roc_score = np.mean(roc) #get mean AUC score for the 5 folds
        roc_diff = np.abs(baseline_roc - roc_score)
        score_difference.append(roc_diff) #append the absolute difference between baseline AUC score and calculated AUC score
        #Perform model fit on 1 column reduced data
        model.fit(x_train_MissingColumn, y_train)
        model_prediction = model.predict_proba(x_test_MissingColumn)[:,1] #predict probability
        probability_difference_lst = []
        for probability in range(len(baseline_predict)):
            #calculate differences between baseline prediced probability and this model's predicted probability
            probability_difference_lst.append(baseline_predict[i] - model_prediction[i])
        probability_difference.append(np.mean(probability_difference_lst)) #append the average probability difference
    
    return (score_difference, probability_difference)

#Load data
train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year
test_data_2012 = load_data('test_2012.csv')
x_test_2012 = test_data_2012[:,3:]

ID_2008 = test_data[:,0]
ID_2012 = test_data_2012[:,0]
cols_delete = data_reduction(x_train, 0.98)
x_train_reduced = delete_cols(x_train, cols_delete)
print(x_train.shape)
print(x_train_reduced.shape)
x_train_normalized = normalize_data(x_train_reduced)

x_test_reduced = delete_cols(x_test, cols_delete)
print(x_test.shape)
print(x_test_reduced.shape)
x_test_normalized = normalize_data(x_test_reduced)

x_test_2012_reduced = delete_cols(x_test_2012, cols_delete)
print(x_test_2012.shape)
print(x_test_2012_reduced.shape)
x_test_2012_normalized = normalize_data(x_test_2012_reduced)

#Calculate the score_difference and probability_difference using get_accuracy_differences function
(score_difference, probability_difference) = get_accuracy_differences(x_train_normalized, y_train, x_test_normalized)

#Load in all the column names from the training data
with open('train_2008.csv') as csvFile:
    reader = csv.reader(csvFile)
    field_names_list = next(reader)
    
features = field_names_list[3:382]

#Perform the data reduction of column names
for i in sorted(cols_delete, reverse=True):
    del features[i]

#sort the scores and get the indices of the top 10 features
indices = np.arange(0,len(score_difference))
score_difference_sorted, indices_sorted = zip(*sorted(zip(score_difference,indices)))

#Print all the column names for the top 10 features
print(features[indices_sorted[265]])
print(features[indices_sorted[264]])
print(features[indices_sorted[263]])
print(features[indices_sorted[262]])
print(features[indices_sorted[261]])
print(features[indices_sorted[260]])
print(features[indices_sorted[259]])
print(features[indices_sorted[258]])
print(features[indices_sorted[257]])
print(features[indices_sorted[256]])



    

