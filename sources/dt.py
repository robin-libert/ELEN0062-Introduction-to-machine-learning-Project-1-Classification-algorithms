"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary



# (Question 1)

def test_and_plot(title, X, y, m_depth=None):
    """Generate an image of our data and the predictions of the decision tree
    
    Parameters
    ----------
    title : str
        The title we give to our image
        
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
        
    m_depth : int > 0, optional (default = None)
        The maximum depth allowed of our decision tree
    
    """
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf = DecisionTreeClassifier(max_depth=m_depth)
    clf.fit(X_train, y_train)
    plot_boundary(title, clf, X_test, y_test)
    
def score(X,y, depth):
    """Compute the precision score on the test sample and on the training sample
    
    Parameters
    ----------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
        
    m_depth : int > 0, optional (default = None)
        The maximum depth allowed of our decision tree
        
    Return
    ------
    train_score : float 0>=, <=1
    
    test_score : float 0>=, <=1
    
    """
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf =DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    train_score = clf.score(X_train, y_train)
    return train_score, test_score
    
def test_set1_accuracy(max_depth):
    """Compute the accuracy of our model with the first data set"""
    test = []
    for i in range(5):
        X,y = make_data1(2000,i+1)
        tr, te = score(X, y, max_depth)
        test.append(te)
    
    test = np.asarray(test)
    my_mean = np.mean(test)
    my_std = np.std(test)
    return my_mean, my_std

def test_set2_accuracy(max_depth):
    """Compute the accuracy of our model with the second data set"""
    test = []
    for i in range(5):
        X,y = make_data2(2000,i+1)
        tr, te = score(X, y, max_depth)
        test.append(te)
    
    test = np.asarray(test)
    my_mean = np.mean(test)
    my_std = np.std(test)
    return my_mean, my_std
        
    
if __name__ == "__main__":
    #make 2 different dataset
    X, y = make_data1(2000,1)
    X2, y2 = make_data2(2000,1)
    
    #save the results in the folder img_dt
    test_and_plot("img_dt/1DecisionTree",X,y)
    test_and_plot("img_dt/2DecisionTree",X2,y2)
    for i in range(4):
        test_and_plot("img_dt/1DecisionTree"+str(2**i),X,y,2**i)
        test_and_plot("img_dt/2DecisionTree"+str(2**i),X2,y2,2**i)
    
    #print the average test set accuracies over five generations of the datasets along with the standard deviation for each depth.    
    print("Mean and standard deviation of 5 random generated set of type 1 with a tree of depth = None : ", end = '')
    print(test_set1_accuracy(None))
    print("Mean and standard deviation of 5 random generated set of type 2 with a tree of depth = None : ", end = '')
    print(test_set2_accuracy(None)) 
    for i in range(4):
        print("Mean and standard deviation of 5 random generated set of type 1 with a tree of depth = "+ str(2**i) + " : ", end = '')
        print(test_set1_accuracy(2**i))
        print("Mean and standard deviation of 5 random generated set of type 2 with a tree of depth = "+ str(2**i) + " : ", end = '')
        print(test_set2_accuracy(2**i))
    
