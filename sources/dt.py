"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary



# (Question 1)

def test_and_plot(title, X, y, m_depth=None):
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[-1850:]
    y_test = y[-1850:]
    clf = DecisionTreeClassifier(max_depth=m_depth)
    clf.fit(X_train, y_train)
    plot_boundary(title, clf, X_test, y_test)
    print(clf.score(X_test, y_test))


if __name__ == "__main__":
    X, y = make_data1(2000)
    X2, y2 = make_data2(2000)
    
    test_and_plot("img_dt/1DecisionTree",X,y)
    test_and_plot("img_dt/2DecisionTree",X2,y2)
    for i in range(4):
        test_and_plot("img_dt/1DecisionTree"+str(2**i),X,y,2**i)
        test_and_plot("img_dt/2DecisionTree"+str(2**i),X2,y2,2**i)
    
