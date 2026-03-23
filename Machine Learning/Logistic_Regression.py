import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import math
class Logistic_Regression:
    
    def __init__(self,lr=0.1,n_iters=100):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters
        
    def fit(self,X_train,y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X_train,self.weights) + self.bias
            logistic_pred = 1 / (1 + np.exp(-linear_pred))
            
            dw = (1/n_samples) * np.dot(X_train.T,(y_train-logistic_pred))
            db = (1/n_samples) * np.sum(y_train-logistic_pred)
            
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
    
    def predict(self,X_test):
        linear_pred = np.dot(X_test,self.weights) + self.bias
        logistic_pred = 1 / (1 + np.exp(-linear_pred))
        class_pred = [ 0 if y<=0.5 else 1 for y in logistic_pred ]
        return class_pred
    
def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)


if __name__ == "__main__":
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
    
   
    model = Logistic_Regression()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print("Accuracy:",accuracy(preds,y_test))
    
    prediction = model.predict(X)
    figure = plt.figure(figsize=(8,6))
    cmap = plt.get_cmap('viridis')
    plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
    plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
    plt.plot(X,prediction,color="black",linewidth=2,label="Best Fit Line")
    plt.show()