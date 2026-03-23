from Decision_Tree import Decision_Tree
import numpy as np
import pandas as pd
from collections import Counter

class Random_Forest:
    def __init__(self,task = 'classification', n_trees = 10,max_depth = 10,min_sample_split = 2,n_feature = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_feature = n_feature
        self.trees = []
        self.task = task
        
    def fit(self,X,y):
        for _ in range(self.n_trees):
            dt = Decision_Tree(max_depth = self.max_depth, 
                               min_sample_split = self.min_sample_split, 
                               n_features = self.n_feature,
                               task=self.task)
            
            X_sampled, y_sampled = self.create_samples(X,y)

            dt.fit(X_sampled,y_sampled)
            self.trees.append(dt)

        
    def create_samples(self,X,y):
        n_samples = X.shape[0]
        indxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[indxs], y[indxs]
    
    
    def majority_vote(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self,X):
        if self.task == "classification":
            tree_preds = np.array([tree.predict(X) for tree in self.trees])
            tree_preds = np.swapaxes(tree_preds,0,1) # 
            predictions = np.array([self.majority_vote(tree_pred) for tree_pred in tree_preds])
            return predictions

        else:
            tree_preds = np.array([tree.predict(X) for tree in self.trees])
            tree_preds = np.swapaxes(tree_preds,0,1) # 
            predictions = np.array([np.mean(tree_pred) for tree_pred in tree_preds])
            return predictions
                    

def accuracy_score(y_true,y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

def mse(y_true,preds):
    return np.mean((y_test - preds)**2)

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    #* Classification 
    X, y = load_breast_cancer(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    rf = Random_Forest(n_trees = 10)
    rf.fit(X_train,y_train)
    preds = rf.predict(X_test)
    print("Classification")
    print("Accuracy:",accuracy_score(y_test,preds))
    
    print('\n')
    
    #* Regression
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    X_train, X_test, y_train, y_test = train_test_split(data,target,test_size = 0.2, random_state = 42)
    rf = Random_Forest(n_trees = 10, task = 'regression')
    rf.fit(X_train,y_train)
    preds = rf.predict(X_test)
    print("Regression")
    print("Mean Squared Error:",mse(y_test,preds))