import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, mode='classification', epsilon=0.1):
        """
        Initialization of SVM model with hyperparameters.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        lambda_param (float): The regularization parameter.
        n_iters (int): Number of iterations for training.
        mode (str): 'classification' or 'regression'.
        epsilon (float): Epsilon value for epsilon-SVM regression.
        """
        if mode not in ['classification', 'regression']:
            raise ValueError("Mode must be 'classification' or 'regression'")
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mode = mode
        self.epsilon = epsilon

    def fit(self, X, y):
        """
        Fits the SVM model to the training data.

        Parameters:
        X (numpy.ndarray): Training feature data.
        y (numpy.ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.mode == 'classification':
            y_ = np.where(y <= 0, -1, 1)
        else:
            y_ = y.copy()
        
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                if self.mode == 'classification':
                    
                    #* Cost Function: Regularized hinge loss function
                    #*                              n 
                    #* J = λ * ||w||^2 + 1/n (Σ max(0, 1 - Y(wx - b)))
                    #*                             i=1 
                    
                    condition = y_[idx] * (np.dot(x, self.weights) - self.bias) >= 1
                    if condition:
                    
                        #* J = λ * ||w||^2
                        #* ∇J = 2λw 
                        #* w = w - α(∇J)
                        #* b = b - α(0)
                        
                        self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                        self.bias = self.bias - self.lr * 0
                        
                    else:
                        #* J = λ * ||w||^2 + 1 - Y(wx - b) 
                        #* ∇J = 2λw - yx
                        #* w = w - α(∇J) 
                        #* b = b - α(Y) 
                        
                        self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x, y_[idx]))
                        self.bias -= self.lr * y_[idx]
                else:
                    #* Cost Function: Regularized epsilon-insensitive loss function
                    #*                              n 
                    #* J = λ * ||w||^2 + 1/n (Σ max(0, |Y - (wx - b)| - ε))
                    #*                             i=1 

                    condition = np.abs(np.dot(x, self.weights) - self.bias - y_[idx]) <= self.epsilon
                    if condition:
                        #* J = λ * ||w||^2
                        #* ∇J = 2λw
                        #* w = w - α(∇J)
                        #* b = b - α(0)
                         
                        self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                    else:
                        #* J = λ * ||w||^2 + |y - (wx - b)| - ε
                        #* ∇J = 2λw - x * (Y - (w*x))
                        #* w = w - α(∇J)
                        #* b = b - α(Y - wx)
                         
                        self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x, y_[idx] - np.dot(x, self.weights)))
                        self.bias -= self.lr * (y_[idx] - np.dot(x, self.weights))
                        
    def predict(self, X):
        """
        Predicts using the trained SVM model.

        Parameters:
        X (numpy.ndarray): Test feature data.

        Returns:
        numpy.ndarray: Predicted labels or values.
        """
        approx = np.dot(X, self.weights) - self.bias
        if self.mode == 'classification':
            return np.sign(approx)
        else:
            return approx

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred) / len(y_true)

def mean_squared_error(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def visualize_svm(X, y, model, task):
    def get_hyperplane_value(x, w, b, offset=0):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    if task == 'classification':
        x1_1 = get_hyperplane_value(x0_1, model.weights, model.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, model.weights, model.bias, 0)

        x1_1_m = get_hyperplane_value(x0_1, model.weights, model.bias, -1)
        x1_2_m = get_hyperplane_value(x0_2, model.weights, model.bias, -1)

        x1_1_p = get_hyperplane_value(x0_1, model.weights, model.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, model.weights, model.bias, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
    
    elif task == 'regression':
        # Regression prediction line
        y0_1 = get_hyperplane_value(x0_1, model.weights, model.bias)
        y0_2 = get_hyperplane_value(x0_2, model.weights, model.bias)
        ax.plot([x0_1, x0_2], [y0_1, y0_2], "b", label='Regression line')

        # Epsilon margin lines
        y0_1_m = y0_1 - model.epsilon
        y0_2_m = y0_2 - model.epsilon
        ax.plot([x0_1, x0_2], [y0_1_m, y0_2_m], "r--", label='Epsilon margin')

        y0_1_p = y0_1 + model.epsilon
        y0_2_p = y0_2 + model.epsilon
        ax.plot([x0_1, x0_2], [y0_1_p, y0_2_p], "r--")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    #* Classification 
    X, y = datasets.make_blobs(n_samples = 50, n_features = 2, centers = 2, cluster_std=1.05, random_state=44)

    y = np.where(y==0,-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=44)

    classifier = SVM()

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    print('Accuracy: ', accuracy(y_test,predictions))

    visualize_svm(X_train, y_train, classifier, 'classification')
    
    # #* Regression
    X, y = datasets.make_regression(n_samples=100, n_features=2, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the SVM regressor
    regressor = SVM(mode='regression', epsilon=2)

    # Fit the regressor to the training data
    regressor.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = regressor.predict(X_test)

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print('MSE:', mse)
    
    visualize_svm(X_train, y_train, regressor, 'regression')
    