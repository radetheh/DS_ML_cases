# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:47:34 2018

@author: Rade Hajder
"""

#Regression for intro to data science
import pandas as pd
dataset =  pd.core.frame.DataFrame()
dataset =  pd.read_excel("Folds5x2_pp.xlsx")

X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting gradient boosting to the Training set (XGBoosting je najbolji a ovaj mu je blizu)
# u osnovi decission trees
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor.fit(X_train, y_train)

#Predicting the Test results
y_pred = regressor.predict(X_test)

# Izgleda da nije neophodno skalirati promenljive vec se koriste takve kakve jesu