# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:22:26 2018

@author: Rade Hajder
"""

#Classification for intro to data science
import pandas as pd
dataset =  pd.core.frame.DataFrame()
dataset =  pd.read_csv("breast-cancer-wisconsin.csv")

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting gradient boosting to the Training set (XGBoosting je najbolji a ovaj mu je blizu)
# u osnovi decission trees
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

#Predicting the Test results
y_pred = classifier.predict(X_test)

# Making confusion matrix
# pravimo pregled false positive i false negativa
# u ovom slucaju su false negativi veoma vazni pa ih treba znati
# scikit learn ima modul metrics koji proracunava tacnost modela (p, sigma etc)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test,y_pred)
