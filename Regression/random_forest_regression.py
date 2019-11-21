#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:40:44 2019

@author: felipe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.DataFrame(pd.read_csv("Position_Salaries.csv"))
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, [2]].values

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=1000, random_state=0)
forest_reg.fit(X, y)

y_pred = forest_reg.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, forest_reg.predict(X_grid), color='blue')
plt.xlabel("Posição")
plt.ylabel("Salário")
plt.title("Salário dos funcionários da empresa")
