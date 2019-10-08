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
import matplotlib.pyplot as plt

from data import make_data1, make_data2
from plot import plot_boundary


# (Question 2)

def test_and_plot(title, X, y, n_neighbors):
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)
    plot_boundary(title, clf, X_test, y_test)
    print(clf.score(X_test, y_test))
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    print(np.mean(scores))
    
def cross_validation(X, y, cv, n_neighbors):
    scores = []
    for i in range(n_neighbors):
        clf = KNeighborsClassifier(i+1)
        scores.append(np.mean(cross_val_score(clf, X, y, cv=10)))

    plt.plot(range(1,151), scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated accuracy')
    
    


if __name__ == "__main__":
    X, y = make_data1(2000)
    X2, y2 = make_data2(2000)
    
    cross_validation(X,y,10,150)
    #cross_validation(X2,y2,10)
    
    test_and_plot("img_knn/1KNN1",X,y,1)
    test_and_plot("img_knn/2KNN1",X2,y2,1)
    test_and_plot("img_knn/1KNN5",X,y,5)
    test_and_plot("img_knn/2KNN5",X2,y2,5)
    test_and_plot("img_knn/1KNN10",X,y,10)
    test_and_plot("img_knn/2KNN10",X2,y2,10)
    test_and_plot("img_knn/1KNN100",X,y,100)
    test_and_plot("img_knn/2KNN100",X2,y2,100)
    test_and_plot("img_knn/1KNN150",X,y,150)
    test_and_plot("img_knn/2KNN150",X2,y2,150)
    test_and_plot("img_knn/1KNN1200",X,y,1200)
    test_and_plot("img_knn/2KNN1200",X2,y2,1200)
