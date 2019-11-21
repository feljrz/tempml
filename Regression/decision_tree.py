#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:25:27 2019

@author: felipe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.DataFrame(pd.read_csv('Position_Salaries.csv'))
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:, [2]].values


#Regressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(X, y)



#Este valor não é bom pois a decision tree encontra a média dos pontos em que separou não é um método linear, portando  apresentará uma linha reta, teste com 6.52
y_pred = tree_reg.predict([[6.5]])

#Ploting
#Ao primeiro momento parece razoavel, porém ao aumentarmos a densidade é observado o real funcionamento 
plt.scatter(X, y, color='red')
plt.plot(X, tree_reg.predict(X), color='blue')
plt.xlabel("Posição")
plt.ylabel("Salário")
plt.title("Salário dos funcionários da empresa")

#Real forma do gráfico
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, tree_reg.predict(X_grid), color='blue')
plt.xlabel("Posição")
plt.ylabel("Salário")
plt.title("Salário dos funcionários da empresa")





