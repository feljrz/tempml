import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.DataFrame(pd.read_csv('Position_Salaries.csv'))
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:, [2]].values

#É nescessário ajustar a escala
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


from sklearn.svm import SVR
reg_svr = SVR(kernel = 'rbf')
reg_svr.fit(X, y)


#Plotting
plt.scatter(X, y, color='green')
plt.plot(X_grid, reg_svr.predict(X_grid), color='blue')


#Predizendo um valor
y_pred = sc_y.inverse_transform(reg_svr.predict(sc_X.transform(np.array([[6.5]]))))


#Aumentando o passo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))





