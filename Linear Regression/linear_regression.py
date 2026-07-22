import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegression:
    def __init__(self,lr = 0.01,n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        X = np.insert(X,0,1,axis=1)
        self.weights = np.zeros(X.shape[1])
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights)
            weights_der = -2 * X.T.dot(y - X.dot(self.weights))
            self.weights = self.weights - (self.lr * weights_der)
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
    
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias        
        