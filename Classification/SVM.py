import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.DataFrame(pd.read_csv("Social_Network_Ads.csv"))
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, [4]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(min(X_set[:,0]) - 1, max(X_set[:, 0]) + 1, 0.01),
                     np.arange(min(X_set[:,0]) - 1, max(X_set[:, 0]) + 1, 0.01))

#verificar countourf e ravel T é transposta
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(("red", "green")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title("Logistic Regression (Test set)")
plt.xlabel("Idade")
plt.ylabel("Salario estimado")
plt.legend()
plt.show()




































 









