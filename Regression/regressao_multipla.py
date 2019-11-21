#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 05:57:58 2019

@author: root
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.DataFrame(pd.read_csv("50_Startups.csv"))
dataset = pd.read_csv("50_Startups.csv")

y_df = pd.DataFrame(dataset.iloc[:,4])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
X_df = pd.DataFrame(X)


dataframe.info() # informações do dataframe
dataframe.shape #forma do dataframe
dataframe.isnull().sum() *100/ dataframe.shape[0] # verificando nulos


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder(categorical_features = [3])
X = one_hot_encoder.fit_transform(X).toarray()

#trap dummyvariable

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#Backward elimination, deixar as variáveis com menor P-value 
#Iremos adicionar uma coluna de 1 pois esta corresponde à X0 

import statsmodels.api as sm

X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X , axis=1)

Xopt = X[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog = y, exog = Xopt).fit()

regressorOLS.summary()


regressorOLS = sm.OLS(endog = y, exog = Xopt).fit()
regressorOLS.summary()

pmaximo = max(regressorOLS.pvalues)

def backwardelimination(x, sl):
    numVariaveis = len(x[0])
    for var in range(numVariaveis):
        regressorOLS = sm.OLS(endog = y, exog = x ).fit()
        pmaximo = max(regressorOLS.pvalues).astype(float)
        if pmaximo > sl:
            for k in range(numVariaveis - var):
                if (regressorOLS.pvalues[k].astype(float) == pmaximo):
                    x = np.delete(x, k, 1)
    
    regressorOLS.summary()
    return x

sl = 0.05
Xopt = X[:, [0, 1, 2, 3, 4, 5]]
Xmodelado = backwardelimination(Xopt, sl)







        
            
        
        
    














