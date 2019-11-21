#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:41:36 2019

@author: felipe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.DataFrame(pd.read_csv("Position_Salaries.csv"))

dataset.info() #mostras as informações do dataset
dataset.isnull().sum()*100  #verificando se há nulos no dataset
dataset.describe()

#verificar se existem outliers no datset

plt1 = sns.boxplot(dataset['Level'])
plt2 = sns.boxplot(dataset['Salary'])

plt.plot(dataset['Level'], dataset['Salary'], color = 'red')

plt.scatter(x= dataset['Level'], y=dataset['Salary'], color = 'blue')

sns.boxplot(x = dataset['Level'], y = dataset['Salary'])

#Não esquecer de importar como matriz
#X = dataset.iloc[:, 1:2].values
X = dataset.iloc[:,[1]].values 
y = dataset.iloc[:, [2]].values

"""#Utilizando o StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
sc_X.fit_transform(X)
sc_y.fit_transform(y)"""

#Suponhamos que seja nescessário utilizar um dummy variable na primeira variável
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoderX = LabelEncoder()
exemplo_labelEncoder = label_encoderX.fit_transform(dataset['Position'])
exemplo_labelEncoder = np.array(exemplo_labelEncoder)
exemplo_labelEncoder = exemplo_labelEncoder.reshape((len(exemplo_labelEncoder), 1))
one_hot_encoderX = OneHotEncoder()
exemplo_OHE = one_hot_encoderX.fit_transform(exemplo_labelEncoder).toarray()"""


"""Esta é uma regressão normal, o que faremos é preparar a matriz com o indices
Constante, X^1 X² ... X^n e colocar no modelo linear
"""

from sklearn.linear_model import LinearRegression #para exemplo
reg_lin1 = LinearRegression()
reg_lin1.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
reg_poly = PolynomialFeatures(degree=4)
X_poly = reg_poly.fit_transform(X)
reg_lin2 = LinearRegression()
reg_lin2.fit(X_poly, y)

#Aumentando o passo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

#Plotting 
plt.scatter(X, y, color= 'green')
plt.plot(X, reg_lin1.predict(X), color = 'blue') #Para exemplificação
plt.plot(X_grid, reg_lin2.predict(reg_poly.fit_transform(X_grid)), color = 'red')
plt.title("Distribuição Salarial e Predições")
plt.xlabel("Nível")
plt.ylabel("Salário")

#Predizendo um valor único
reg_lin2.predict(reg_poly.fit_transform([[6.5]]))


















