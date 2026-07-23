from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionGD

X,y = make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,random_state=41,hypercube=False,class_sep=20)



model = LogisticRegressionGD()
intercept,coef = model.fit(X,y)

m = -(coef[0]/coef[1])
b = -(intercept/coef[1])

x_input = np.linspace(-3,3,100)
y_input = m*x_input + b


plt.plot(x_input,y_input,color ='red',linewidth = 3)
plt.scatter(X[:,0],X[:,1],c = y)
plt.ylim(-3,2)
plt.show()