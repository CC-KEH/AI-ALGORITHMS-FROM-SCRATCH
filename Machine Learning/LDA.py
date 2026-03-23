import numpy as np


class LDA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        labels = np.unique(y)

        # Within Class
        mean_all = np.mean(X, axis=0)
        sw = np.zeros((n_features, n_features))
        sb = np.zeros((n_features, n_features))
        for c in labels:
            xc = X[c == y]
            mean_c = np.mean(xc, axis=0)

            # (4,features) * (features * 4) => (4,4)
            sw += (xc - mean_c).T.dot(xc - mean_c)
            nc = xc.shape[0]

            # (4,1) * (4,1)T => (4,4)
            mean_diff = (mean_c - mean_all).reshape(n_features, 1)
            sb += nc * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(sw).dot(sb)

        # Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T  # Because we transposed earlier
        idxs = np.argsort(abs(eigenvalues))[::-1] # Descending Order
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[: self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    data = load_iris()
    X, y = data.data, data.target

    lda = LDA(n_components=2)
    lda.fit(X, y)
    x_projected = lda.transform(X)

    print("Shape of X before transforming:", X.shape)
    print("Shape of X after transforming:", x_projected.shape)

    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("linear discriminant 1")
    plt.ylabel("linear discriminant 2")
    plt.colorbar()
    plt.show()
