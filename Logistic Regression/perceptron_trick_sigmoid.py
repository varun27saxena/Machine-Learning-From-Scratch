import numpy as np
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 /(1+np.exp(-z))

class Perceptron:
    def __init__(self,n_iter = 1000,lr = 0.1):
        self.weights = None
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self,X,y):
        X = np.insert(X,0,1,axis = 1)
        self.weights = np.ones(X.shape[1])
        for i in range(self.n_iter):
            j = np.random.randint(0,100)
            y_pred = sigmoid(np.dot(X[j],self.weights))
            self.weights = self.weights + self.lr*(y[j]-y_pred)*X[j]    
        return self.weights[0],self.weights[1:]    