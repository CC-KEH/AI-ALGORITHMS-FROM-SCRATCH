import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Node:
    
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    

class Decision_Tree:
    
    def __init__(self, max_depth=10, min_sample_split=2, criteria='entropy', n_features=None):
        self.max_depth = max_depth                # Stopping Criteria
        self.min_sample_split = min_sample_split  # Stopping Criteria
        self.criteria = criteria                  # Criteria type, entropy in our case.
        self.n_features = n_features              # No of features we'll be using for constructing the tree.
        self.root = None
        
    def construct_tree(self, X, y, n_features, n_samples, depth=0):
        # No of labels there are in a specific feature,
        # if 1 then no need to split. 
        # For eg. in case of Wind, we can go to 2 labels (strong and weak) so we split.
        labels = len(np.unique(y))  

        # Check the stopping criteria 
        # if met create a leaf node, based on label with max frequence
        if depth >= self.max_depth or labels == 1 or n_samples < self.min_sample_split:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feat_indexs = np.random.choice(n_features, self.n_features, replace=False)
        best_threshold, best_feature = self.best_split(X, y, feat_indexs)

        # Create child nodes (Recursively Create Tree)
        left_indxs, right_indxs = self.split(X[:, best_feature], best_threshold)
        left = self.construct_tree(X[left_indxs, :], y[left_indxs], n_samples, depth + 1)
        right = self.construct_tree(X[right_indxs, :], y[right_indxs], n_samples, depth + 1)
        return Node(best_feature, best_threshold, left, right)
        
    def most_common_label(self,y):
        c = Counter(y)
        return c.most_common(1)[0][0]
        
    def calculate_entropy(self, y):
        hist = np.bincount(y) # returns a frequency list of elements, from 0 to max(y).
        ps = hist / len(y)    # [p(x1), p(x2), p(x3),..., p(xN)]
        return -np.sum([p * np.log(p) for p in ps if p > 0])  # Only consider non-zero probabilities

    def split(self,X_col,threshold):
        # Left Split, val <= threshold
        left_idxs  = np.argwhere(X_col<=threshold).flatten()
        # Right Split, val> threshold
        right_idxs = np.argwhere(X_col>threshold).flatten()
        # np.argwhere returns the indices in a list of lists, so we flatten the result.
        return left_idxs, right_idxs
        
    def calculate_gain(self,X_col,y,threshold):
        # IG = E(parent) - [weighted average] * E(children)
        
        # Calculate entropy of the parent 
        entropy_parent = self.calculate_entropy(y)
        
        # Create children
        left_idxs,right_idxs = self.split(X_col,threshold)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
            
        # Calculate the weighted average entropy of the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.calculate_entropy(y[left_idxs]), self.calculate_entropy(y[right_idxs])
        
        # No of samples in left/total samples times left entropy + No of samples in right/total samples times right entropy.
        weighted_average_entropy_children = (n_l/n) * e_l + (n_r/n) * e_r
        
        # Calculate Information Gain
        info_gain = entropy_parent - weighted_average_entropy_children
        
        return info_gain 
            
    def best_split(self,X,y,feat_indexs):
        best_gain = -1
        split_index = None
        split_threshold = None
        
        for feat_index in feat_indexs:
            X_col = X[:, feat_index]       # Values of feature X_col.
            thresholds = np.unique(X_col)  # Selecting all the unique values of X_col as thresholds.
            
            for thr in thresholds:
                # For each threshold in thresholds.
                # Find the threshold with maximum Information Gain
                gain = self.calculate_gain(X_col,y,thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_index
                    split_threshold = thr
                
        return split_threshold, split_index 
          
    def traverse_tree(self, X, node):
        if node.is_leaf():
            return node.value

        if X[node.feature] <= node.threshold:       # Recursively travel
            return self.traverse_tree(X, node.left) # val <= threshold, Goto Left child.
        return self.traverse_tree(X, node.right)    # val > threshold, Goto Right child.        
                 
    def fit(self, X_train, y_train):
        self.n_features = X_train.shape[1] if not self.n_features else min(X_train.shape[1], self.n_features)   # So that the no of features in the tree do not exceed the actual no of features we have in data.
        n_samples = X_train.shape[0]  # No of samples.
        self.root = self.construct_tree(X_train, y_train, self.n_features, n_samples)

    def predict(self,X):
        return np.array([self.traverse_tree(x,self.root) for x in X])
    
def mse(y1,y2):
    return np.mean((y2-y1)**2)

def accuracy(y1,y2):
    return np.sum(y1==y2)/len(y1)

if __name__ == "__main__":
    model = Decision_Tree()
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    error = mse(y_test,preds)
    print(f"Accuracy: {accuracy(y_test,preds)}")
    print('MSE:',error)