#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:22:06 2019

@author: evascheller
"""

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

#Here create a loop that fits randomforest to multiple different PCA dimensions
#Plot the dimension number versus test score to find sweet-spot number of dimensions
dimension_lst = np.arange(50,len(x_train_reduced[0]),10)
print(len(dimension_lst))
cv_for_dimension_lst = []
ROC_AUC_for_dimension_lst = []
n=1
for dimensions in dimension_lst:
    print('update {}'.format(n))
    x_new_dimension = Dimensionality_reduction_PCA(x_train_reduced, dimensions)
    model = RandomForestClassifier(criterion='gini')
    (cv_accuracy, roc_auc_scores) = cross_validating_randomforest(model, x_new_dimension, y_train)
    cv_for_dimension_lst.append(cv_accuracy)
    ROC_AUC_for_dimension_lst.append(np.average(roc_auc_scores))
    n+=1
    
plt.figure(1)
plt.plot(dimension_lst,ROC_AUC_for_dimension_lst)
plt.savefig('dimensionalityVSAUC')
plt.show()