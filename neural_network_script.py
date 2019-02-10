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
from keras.utils import to_categorical

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

# for each column, divide all of it by its max value
def normalize_data(x_data):
    new_x = x_data.copy()
    shape = new_x.shape
    for i in range(shape[1]):
        col = new_x[:,i]
        maxVal = np.max(col)
        new_x[:,i] /= maxVal
    return new_x


def cross_val_NN(model, x_data, y_data):
    kf = KFold(n_splits=5)
    training_accuracy = []
    test_accuracy = []
    weights = model.get_weights()
    for train_index, test_index in kf.split(x_data):
        model.set_weights(weights)
        x_train_fold, x_test_fold = x_data[train_index], x_data[test_index]
        y_train_fold, y_test_fold = y_data[train_index], y_data[test_index]
        training_model = model.fit(x_train_fold, y_train_fold, epochs=10, batch_size=1000)
        training_accuracy.append(model.evaluate(x=x_train_fold, y=y_train_fold)[1])
        test_accuracy.append(model.evaluate(x=x_test_fold,y=y_test_fold)[1])
    training_accuracy = np.array(training_accuracy)
    test_accuracy = np.array(test_accuracy)

    return (training_accuracy, test_accuracy)

train_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')
y_train = train_data[:,382]
x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
x_test = test_data[:,3:] #Here I remove the first 3 columns representing ID, month, and year

x_train_reduced = data_reduction(x_train, 0.98)
print(x_train.shape)
print(x_train_reduced.shape)

print(x_train_reduced)
x_train_reduced = normalize_data(x_train_reduced)

input_size = x_train_reduced.shape[1]
y_train_binary = to_categorical(y_train)

# create model
model = Sequential()
model.add(Dense(input_size, input_shape=(input_size,)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.3))

model.add(Dense(200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Cross val to get an idea of accuracies.
train_acc, test_acc = cross_val_NN(model, x_train_reduced, y_train_binary)

print('training_accuracy {}'.format(np.mean(train_acc)))
print('test_accuracy {}'.format(np.mean(test_acc)))
