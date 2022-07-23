# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:10:05 2022

@author: bilal
"""

from sklearn.datasets import make_circles
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
#import matplotlib.pyplot as plt

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)


kf = KFold(n_splits=5, shuffle=True, random_state=1)
lr_scores = []
rf_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train)
    lr_scores.append(lr.score(X_test, y_test))
    
    
    param_grid = {'n_estimators': [x for x in range(1,101)]}
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, param_grid, scoring='f1')
    gs.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=list(gs.best_params_.values())[0])
    rf.fit(X_train, y_train)
    rf_scores.append(rf.score(X_test, y_test))
    
    
    
print("LR accuracy:", np.mean(lr_scores))
print("RF accuracy:", np.mean(rf_scores))