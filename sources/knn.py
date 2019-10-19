"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from data import make_data1, make_data2
from plot import plot_boundary
import matplotlib.pyplot as plt


# (Question 2)

def test_and_plot(title, X, y, n_neighbors):
    """Generate an image of our data and the predictions of the decision tree
    
    Parameters
    ----------
    title : str
        The title we give to our image
        
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
        
    n_neighbors : int > 0, optional (default = None)
        The number of neighbors of our model.
    
    """
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)
    plot_boundary(title, clf, X_test, y_test)
    
def cross_validation(X, y, n_neighbors):
    """Compute the 10-fold cross-validation score with our model and a number of neighbors goes from 1 to n_neighbors"""
    scores = []
    for i in range(n_neighbors):
        clf = KNeighborsClassifier(i+1)
        cv = cross_val_score(clf, X[:150], y[:150], cv = 10)
        scores.append(np.mean(cv))
    plt.plot(range(1,n_neighbors+1), scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated accuracy')
    plt.savefig("img_knn/cross_validation.pdf")
    return scores
    
def optimal_number_of_neighbors(X,y):
    """Return the optimal number of neighbors for the dataset X, y"""
    scores = cross_validation(X,y, 134)
    return scores.index(max(scores))+1

def optimal_precision(X,y,optimal):
    """Compute the precision of our model over the dataset X, y when we use the optimal number of neighbors"""
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf = KNeighborsClassifier(optimal)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

if __name__ == "__main__":
    #make 2 different dataset
    X, y = make_data1(2000, 1)
    X2, y2 = make_data2(2000, 1) 
    
    #save the results in the folder img_knn
    test_and_plot("img_knn/1KNN1",X,y,1)
    test_and_plot("img_knn/2KNN1",X2,y2,1)
    test_and_plot("img_knn/1KNN5",X,y,5)
    test_and_plot("img_knn/2KNN5",X2,y2,5)
    test_and_plot("img_knn/1KNN10",X,y,10)
    test_and_plot("img_knn/2KNN10",X2,y2,10)
    test_and_plot("img_knn/1KNN75",X,y,75)
    test_and_plot("img_knn/2KNN75",X2,y2,75)
    test_and_plot("img_knn/1KNN100",X,y,100)
    test_and_plot("img_knn/2KNN100",X2,y2,100)
    test_and_plot("img_knn/1KNN150",X,y,150)
    test_and_plot("img_knn/2KNN150",X2,y2,150)
    
    #Find the optimal number of neighbors for our model with the training set of our second dataset
    print("Optimal number of neighbors = ", end='')
    optimal = optimal_number_of_neighbors(X2,y2)
    print(optimal)
    
    #compute the accuracy over the second dataset when we use our model with an optimal number of neighbors
    print("Accuracy of the second test set with the optimal number of neighbors = ", end='')
    print(optimal_precision(X2,y2,optimal))
