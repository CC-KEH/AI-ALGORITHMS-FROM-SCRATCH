import numpy as np


class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance, we transpose because func needs samples as columns
        cov = np.cov(X.T)

        # Eigenvectors, Eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # Eigenvectors v is a column vector, So we transform this for easier calculation
        eigenvectors = eigenvectors.T

        # Sort Eigenvalues, in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Selecting our n_components
        self.components = eigenvectors[: self.n_components]

    def transform(self, X):
        # Projects data
        X = X - self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar()
    plt.show()
