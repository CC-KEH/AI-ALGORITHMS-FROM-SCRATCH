import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Linear_Regression: 
    def __init__(self,lr=0.1,n_iters=100):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            '''
            Here, 
            fs are features
            
                 f1  f2   f3  f4  f5  
            X = [x11,x12,x13,x14,x15]  weights = [w1]  bias = bias
                [x21,x22,x23,x24,x25]            [w2]
                [x31,x32,x33,x34,x35]            [w3]
                [x41,x42,x43,x44,x45]            [w4]
                [x51,x52,x53,x54,x55]            [w5]
            '''
            y_pred = np.dot(X,self.weights) + self.bias         
        
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
        
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
        
    def predict(self,X_test):
        predicted = np.dot(X_test,self.weights) + self.bias
        return predicted

def mse(y1,y2):
    return np.mean((y2-y1)**2)

if __name__ == "__main__":
    X,y = datasets.make_regression(n_samples=300,n_features=1,noise=10,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
    
    model = Linear_Regression()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print("Mean Squared Error:",mse(preds,y_test))
    
    fig = plt.figure(figsize=(8,6))
    predictions = model.predict(X)
    cmap = plt.get_cmap('viridis')
    plt.scatter(X_train,y_train,color=cmap(0.9),s=10,label='Training Data')
    plt.scatter(X_test,y_test,color=cmap(0.5),s=10,label='Test Data')
    plt.plot(X,predictions,color="black",linewidth=2,label="Best Fit Line")
    plt.legend()
    plt.show()