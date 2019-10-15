"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    mean_X = np.array([])
    var_X = np.array([])
    p_y = np.array([])
    ppredict_y0 = np.array([])
    ppredict_y1 = np.array([])


    def fit(self, X, y):
        """Fit a Gaussian navie Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================
        
        #indexes of y when y = 0 and when y = 1
        y_0 = np.where(y==0)
        y_1 = np.where(y==1)
        #mean of the value of the two input variables when y = 0 and when y = 1
        mean_X0 = np.array([np.mean(X[y_0,0]), np.mean(X[y_0,1])])
        mean_X1 = np.array([np.mean(X[y_1,0]), np.mean(X[y_1,1])])
        self.mean_X = np.array([mean_X0, mean_X1])
        #variance of the value of the two input variables when y = 0 and when y = 1
        var_X0 = np.array([np.var(X[y_0,0]),np.var(X[y_0,1])])
        var_X1 = np.array([np.var(X[y_1,0]),np.var(X[y_1,1])])
        self.var_X = np.array([var_X0,var_X1])
        #[probability y = 0, probability y = 1] 
        self.p_y = np.array([np.size(y_0)/np.size(y), np.size(y_1)/np.size(y)])

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================
        
        p_xi_y0 = 1./np.sqrt(2*np.pi*self.var_X[0]**2)*np.exp(-(((X-self.mean_X[0])**2)/(2*self.var_X[0])))
        p_xi_y1 = 1./np.sqrt(2*np.pi*self.var_X[1]**2)*np.exp(-(((X-self.mean_X[1])**2)/(2*self.var_X[1])))
        self.ppredict_y0 = self.p_y[0]*p_xi_y0[:,0]*p_xi_y0[:,1]
        self.ppredict_y1 = self.p_y[1]*p_xi_y1[:,0]*p_xi_y1[:,1] 
        
        predict_y = []
        for i in range(len(self.ppredict_y0)):
            if self.ppredict_y0[i] < self.ppredict_y1[i]:
                predict_y.append(1)
            else:
                predict_y.append(0)
        return np.array(predict_y)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================
        return np.array([self.ppredict_y0,self.ppredict_y1])

if __name__ == "__main__":

    X, y = make_data1(2000,1)
    X2, y2 = make_data2(2000, 1)
    clf = GaussianNaiveBayes()
    clf = clf.fit(X[:150],y[:150])
    p = clf.predict(X[-1850:])
    
    clf2 = GaussianNaiveBayes()
    clf2 = clf.fit(X2[:150],y2[:150])
    p2 = clf.predict(X2[-1850:])
    
    print(accuracy_score(y[-1850:], p))
    print(accuracy_score(y2[-1850:], p2))