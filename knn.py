import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, isclassifier, k=3):
        self.k = k
        self.isclassifier = isclassifier
        
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
        
    def predict(self,X):
        self.x_test = X
        predictions = [self._predict_single(x) for x in X]
        return predictions
    
    def _predict_single(self,x1):
        # Find distance between x1 and all other points of x_train
        distances = [euclidean_distance(x1,x2) for x2 in self.x_train]
        # Sort the distances, and get the index of top k points closest to x1.
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_nbrs = [self.y_train[i] for i in k_indices]
        
        if self.isclassifier:
            # most_common: returns a list of tuples
            # [(x1,freq(x1),(x2,freq(x2),...,(xn,freq(xn)]
            prediction = Counter(k_nearest_nbrs).most_common() 
            return prediction[0][0]
        else:
            # Find average of k nearest neighbours
            return np.mean(k_nearest_nbrs)


if __name__ == "__main__":
    cmap = ListedColormap(["#FF0000","#00FF00","#0000FF"])
    
    #* Classification
    # Generate a synthetic classification dataset
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=44)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # plt.figure()
    # plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolors='k',s=20)
    # plt.show()
    
    classifier = KNN(isclassifier=True,k=5)
    classifier.fit(x_train,y_train)
    preds = classifier.predict(x_test)
    
    accuracy = np.sum(preds==y_test) / len(y_test)
    print("On Classification Task")
    print("Accuracy:", accuracy)
    
    #* Regression 
    # Generate a synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # plt.figure()
    # plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolors='k',s=20)
    # plt.show()
    
    # Fit the KNN regressor to the training data
    regressor = KNN(isclassifier=False, k=5)
    regressor.fit(x_train, y_train)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((y_test - regressor.predict(x_test))**2))
    print("On Regression Task")
    print("RMSE:", rmse)
