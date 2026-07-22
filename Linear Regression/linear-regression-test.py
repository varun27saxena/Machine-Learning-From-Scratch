from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error


X,y = make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# plt.scatter(X[:,0],y)
# plt.show()

print(X_train.shape)

lr = LinearRegression(lr=0.01)

lr.fit(X_train,y_train)

print(lr.bias)
print(lr.weights)

y_pred = lr.predict(X_test)

print(f"mean absolute error {mean_absolute_error(y_test,y_pred)}")
print(f"mean squared error {mean_squared_error(y_test,y_pred)}")
print(f"root mean squared error {root_mean_squared_error(y_test,y_pred)}")
print(f"R-2 score {r2_score(y_test,y_pred)}")

plt.scatter(X[:,0],y)
plt.plot(X_train,lr.predict(X_train),color = 'r')
plt.show()

