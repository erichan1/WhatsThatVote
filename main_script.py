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

def write_file(filename, ID, target):
    '''
    This function writes a csv file for submission purposes
    Input: 
        filename: name of file
        ID: an array of id numbers
        target: the predicted probability of the positive class
    '''
    new_ID, new_target = zip(*sorted(zip(ID,target))) #sort the id and target according to id numbers
    new_ID_array = []
    for value in list(new_ID):
        new_ID_array.append(int(value)) #transform ID numbers to integers instead of float
        
    with open(filename,'w') as f: #write the csv file
        f.write('id,target\n')
        writer=csv.writer(f,delimiter=',')
        writer.writerows(zip(new_ID_array,list((new_target)))) #write id and target in separate columns
    f.close()
        
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

x_test_reduced = delete_cols(x_test, cols_delete) #dele columns for 2008 test data
print(x_test.shape)
print(x_test_reduced.shape)
x_test_normalized = normalize_data(x_test_reduced) #normalize 2008 test data

x_test_2012_reduced = delete_cols(x_test_2012, cols_delete) #dele columns for 2012 test data
print(x_test_2012.shape)
print(x_test_2012_reduced.shape)
x_test_2012_normalized = normalize_data(x_test_2012_reduced) #normalize 2012 test data

#Potential further reduction using GettingBadData script
#Here remove any features that resulted in a negative influence of the AUC score from the GettingBadData script
HighAUCColumns = [77,190,9,98,121,138,161,28,13,172,249,130,199,7,233,238,241,2,136,205,247,239,127,119,47,96,12,74,37,31]
LowAUCColumns = [64,
 125,
 120,
 128,
 245,
 10,
 213,
 183,
 169,
 260,
 207,
 206,
 110,
 100,
 86,
 112,
 83,
 194,
 174,
 218,
 147,
 177,
 209,
 198,
 90,
 97,
 167,
 154,
 158,
 41,
 142,
 228,
 185,
 25,
 148,
 160,
 188,
 70,
 93,
 107,
 235,
 ]

other_LowAUCColumns = [192,
 60,
 254,
 139,
 30,
 134,
 61,
 265,
 68,
 242,
 173,
 124,
 39,
 8,
 191,
 210,
 232,
 212,
 251,
 57,
 166,
 168,
 36,
 99,
 62,
 215,
 253,
 111,
 69,
 146,
 229,
 129,
 33,
 170,
 122,
 88,
 214,
 155,
 180,
 102,
 131,
 76,
 263,
 14,
 175,
 261,
 144,
 151,
 71,
 123,
 257,
 162,
 262,
 48,
 108,
 135,
 178,
 104,
 165,
 227,
 230,
 51,
 182,
 223,
 221,
 18,
 63,
 258,
 17,
 23,
 115,
 50,
 32,
 65,
 16,
 153,
 73,
 55,
 109,
 43,
 133,
 184,
 117,
 4,
 208,
 152,
 42,
 126,
 34,
 6,
 237,
 85,
 200,
 225,
 141,
 75,
 40,
 145,
 140,
 80,
 94,
 181,
 1,
 143,
 105,
 21,
 15,
 176,
 202,
 203,
 195,
 220,
 132,
 243,
 59,
 11,
 219,
 89,
 179,
 189,
 222,
 78,
 56,
 252,
 101,
 248,
 5,
 217,
 26,
 196,
 240,
 211,
 255,
 0,
 49,
 24,
 204,
 87,
 103,
 38,
 22,
 79,
 259,
 159,
 106,
 216,
 53,
 81,
 137,
 224,
 234,
 264,
 19,
 52,
 201]

x_train_NoBadData = delete_cols(x_train_reduced, LowAUCColumns)

#Now perform actual model fit with optimized parameters and dimensions
model = RandomForestClassifier(criterion = 'gini')
model.set_params(n_estimators=1000, max_features='auto', min_samples_leaf=25)
model.fit(x_train_reduced, y_train)
#(cv_accuracy,roc)=cross_validating_randomforest(model, x_train_NoBadData, y_train)
x_test_NoBadData = delete_cols(x_test_reduced, LowAUCColumns)
x_test_2012_NoBadData = delete_cols(x_test_2012_reduced, LowAUCColumns)
target_2008 = model.predict_proba(x_test_NoBadData)[:,1]
target_2012 = model.predict_proba(x_test_2012_NoBadData)[:,1]
#Write files
write_file('2008_probabilities.csv',ID_2008,target_2008)
write_file('2012_probabilities.csv',ID_2008,target_2012)

