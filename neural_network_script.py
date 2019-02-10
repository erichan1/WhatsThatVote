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
    return delete_cols

def delete_cols(dataset, delete_cols):
    return np.delete(dataset, delete_cols, 1)

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
        training_model = model.fit(x_train_fold, y_train_fold, epochs=30, batch_size=1000)
        training_accuracy.append(model.evaluate(x=x_train_fold, y=y_train_fold)[1])
        test_accuracy.append(model.evaluate(x=x_test_fold,y=y_test_fold)[1])
        

    training_accuracy = np.array(training_accuracy)
    test_accuracy = np.array(test_accuracy)

    return (training_accuracy, test_accuracy)

def create_NN(input_size):
    # encapsulate the model I've built so far
    # if you want to play w model parameters do it here
    # may add parameters to this function so I can do sensitivity training

    model = Sequential()
    model.add(Dense(input_size, input_shape=(input_size,)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Dropout(0.4))
    
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def cross_val_test():
    # this isn't really a function. 
    # This encapsulates the code to get cross val in its own block
    # Call this in the main function. 

    train_data = load_data('train_2008.csv')
    test_data = load_data('test_2008.csv')
    y_train = train_data[:,382]
    x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
    
    x_train_reduced = data_reduction(x_train, 0.98)
    print(x_train.shape)
    print(x_train_reduced.shape)
    
    x_train_reduced = normalize_data(x_train_reduced)
    
    input_size = x_train_reduced.shape[1]
    y_train_binary = to_categorical(y_train)
    model = create_NN(input_size)
    
    # Cross val to get an idea of accuracies.
    train_acc, test_acc = cross_val_NN(model, x_train_reduced, y_train_binary)
    
    print('training_accuracy {}'.format(np.mean(train_acc)))
    print('test_accuracy {}'.format(np.mean(test_acc)))

def final_test():
    # also not really a function
    # encapsulates the process of training the final model and using it on test data

    train_data = load_data('train_2008.csv')
    test_data = load_data('test_2008.csv')
    y_train = train_data[:,382]
    x_train = train_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year
    x_test = test_data[:,3:382] #Here I remove the first 3 columns representing ID, month, and year

    ID_2008 = test_data[:,0]
    cols_delete = data_reduction(x_train, 0.98)
    
    x_train_reduced = delete_cols(x_train, cols_delete)
    print(x_train.shape)
    print(x_train_reduced.shape)

    x_test_reduced = delete_cols(x_test, cols_delete)
    print(x_test.shape)
    print(x_test_reduced.shape)
    
    x_train_reduced = normalize_data(x_train_reduced)
    x_test_reduced = normalize_data(x_test_reduced)
    
    input_size = x_train_reduced.shape[1]
    y_train_binary = to_categorical(y_train)
    model = create_NN(input_size)
    
    trained_model = model.fit(x_train_reduced, y_train_binary, epochs=30, batch_size=1000)

    Y_pred = trained_model.predict(x_test)
    ID_2008 = test_data[:,0]

    write_file('2008_probabilities_NN',ID_2008,Y_pred)

if __name__ == '__main__':
    # cross_val_test() 
    final_test()
    
def write_file(filename, ID, target):
    '''
    Insert some text
    '''
    new_ID, new_target = zip(*sorted(zip(ID,target)))
    new_ID_array = []
    for value in list(new_ID):
        new_ID_array.append(int(value))
        
    with open(filename,'w') as f:
        f.write('id,target\n')
        writer=csv.writer(f,delimiter=',')
        writer.writerows(zip(new_ID_array,list((new_target))))
    f.close()
        
#Load data

    
