#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:00:09 2019

@author: evascheller
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers
from sklearn.model_selection import KFold

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



train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year

x_train_reduced = data_reduction(x_train, 0.98)
print(x_train.shape)
print(x_train_reduced.shape)

#Set up model
model = Sequential()
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(len(x_train_reduced)))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Final model for maximum test accuracy
kf = KFold(n_splits=5)
training_accuracy = []
for train_index, test_index in kf.split(x_train_reduced):
    x_train_fold, x_test_fold = x_train_reduced[train_index], x_train_reduced[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    training_model = model.fit(x_train, y_train, epochs=1, batch_size=1000,
                    validation_data=(x_test_fold, y_test_fold))
    training_accuracy.append(model.evaluate(x=x_train_fold, y=y_train_fold)[1])
    test_accuracy.append(model.evaluate(x=x_test_fold,y=y_test_fold)[1])

print('training_accuracy {}'.format(np.mean(training_accuracy)))
print('test_accuracy {}'.format(np.mean(test_accuracy))
    
    





