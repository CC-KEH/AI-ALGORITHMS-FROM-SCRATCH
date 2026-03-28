import numpy as np
import matplotlib.pyplot as plt

#* UTILITY FUNCTIONS
# All methods are static, pure mathematical operations with no object state.
class Utils:

    def ReLU(Z):
        return np.maximum(0, Z)

    def derivative_ReLU(Z):
        """
        Gradient of ReLU, used during backpropagation.
        Returns 1 where Z > 0, and 0 elsewhere (as a boolean array).
        This "gates" the gradient: neurons that were inactive (Z <= 0)
        receive zero gradient and don't update, the "dying ReLU" trade-off.
        """
        return Z > 0

    def softmax(Z):
        """
        Numerical stability fix: subtract max(Z) before exponentiating.
        This prevents overflow (e.g., exp(1000) -> inf) without changing the
        output, because the shift cancels in numerator and denominator.
        """
        shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z   = np.exp(shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def one_hot(Y):
        """
        Converts integer class labels into one-hot encoded vectors.
        Example: label 3 with 10 classes -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        """
        encoded_Y = np.zeros((Y.size, Y.max() + 1))
        encoded_Y[np.arange(Y.size), Y] = 1
        return encoded_Y.T   # -> (num_classes, m)

    def accuracy(y_pred, y):
        """Fraction of predictions that match the true labels."""
        return np.sum(y_pred == y) / y.size

class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate=0.1, iterations=500):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            Neuron count for every layer, including input and output.
            Examples:
              [784, 10]               -> 0 hidden layers (linear classifier)
              [784, 64, 10]           -> 1 hidden layer
              [784, 128, 64, 10]      -> 2 hidden layers

            First value  = number of input features  (784 for MNIST).
            Last value   = number of output classes  (10 for digits 0-9).

        learning_rate : float
            Step size for gradient descent (alpha). Default 0.1.

        iterations : int
            Number of full passes over the training data. Default 500.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 entries (input + output).")

        self.layer_sizes = layer_sizes
        self.alpha = learning_rate
        self.n_iters = iterations
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices
        self.utils = Utils

        # Populated by init_params()
        self.weights = []   # weights[i] shape: (layer_sizes[i+1], layer_sizes[i])
        self.biases = []    # biases[i]  shape: (layer_sizes[i+1], 1)

        # Populated during forward / backward
        self.Z = []   # pre-activation values  per layer
        self.A = []   # post-activation values per layer  (A[0] == X)

    #* PARAMETER INITIALIZATION
    def init_params(self):
        """
        Randomly initialize one weight matrix and one bias vector per layer.

        Weight matrix shape for layer i:  (layer_sizes[i+1], layer_sizes[i])
          - rows    = neurons in the NEXT layer    (each has its own weight row)
          - columns = neurons in the CURRENT layer (inputs fed into the next layer)

        Scaling weights by 0.1 keeps initial values small, which:
          - Prevents ReLU saturation on the very first forward pass.
          - Keeps gradients in a reasonable range from the start.

        Biases start at zero , safe default since weight randomness already
        breaks symmetry between neurons.
        """
        self.weights = []
        self.biases  = []

        for i in range(self.n_layers):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            W = np.random.randn(n_out, n_in) * 0.1                        # small random weights
            b = np.zeros((n_out, 1))      # zero biases

            self.weights.append(W)
            self.biases.append(b)

    #* FORWARD PASS
    def forward(self, X):
        """
        Propagate input X through every layer to produce predictions.

        For each layer i (0-indexed):
          Z[i] = W[i] . A[i] + b[i]    (linear combination)
          A[i+1] = activation(Z[i])    (non-linear transformation)

        Activation choice:
          - All hidden layers -> ReLU    (fast, sparse, works well in practice)
          - Output layer      -> Softmax (turns raw scores into class probabilities)

        We set A[0] = X so the same loop covers every layer uniformly.

        After this call:
          self.A[-1]  holds the final probability matrix, shape (n_classes, m)
        """
        self.Z = [None] * self.n_layers
        self.A = [None] * (self.n_layers + 1)
        self.A[0] = X   # "activation" of the input layer is the raw input

        for i in range(self.n_layers):
            self.Z[i] = self.weights[i].dot(self.A[i]) + self.biases[i]

            # Last layer -> softmax; all earlier layers -> ReLU
            if i == self.n_layers - 1:
                self.A[i + 1] = self.utils.softmax(self.Z[i])
            else:
                self.A[i + 1] = self.utils.ReLU(self.Z[i])

    #* BACKWARD PASS (Backpropagation)
    def backward(self, X, Y):
        """
        Compute gradients of the cross-entropy loss w.r.t. every W and b.
        """
        
        m  = Y.size
        dZ = [None] * self.n_layers
        self.dW = [None] * self.n_layers
        self.db = [None] * self.n_layers

        # --- Output layer gradient (softmax + cross-entropy derivative) ---
        one_hot_Y = self.utils.one_hot(Y)  # (n_classes, m)
        dZ[-1] = self.A[-1] - one_hot_Y                                    # (n_classes, m)

        # --- Backpropagate through all layers, right to left ---
        for i in reversed(range(self.n_layers)):
            # Gradients for this layer's weights and biases
            self.dW[i] = 1 / m * dZ[i].dot(self.A[i].T)
            self.db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)

            # Propagate error further left (skip when we've reached the input)
            if i > 0:
                # Route error back through weight matrix, then gate by ReLU'
                dZ[i - 1] = (
                    self.weights[i].T.dot(dZ[i])
                    * self.utils.derivative_ReLU(self.Z[i - 1])
                )

    #* PARAMETER UPDATE (Gradient Descent)
    def update_params(self):
        """
        Apply one gradient-descent step to every weight matrix and bias vector.
        Rule:  θ <- θ - α * dL/dθ
        """
        
        for i in range(self.n_layers):
            self.weights[i] -= self.alpha * self.dW[i]
            self.biases[i]  -= self.alpha * self.db[i]

    #* TRAINING LOOP
    def train(self, X, Y):
        """
        Full training loop:  init -> (forward -> backward -> update) x n_iters

        X : (n_input_features, m)  - each column is one training example
        Y : (m,)                   - integer class labels 0 ... n_classes-1
        """
        
        self.init_params()

        for i in range(self.n_iters):
            self.forward(X)
            self.backward(X, Y)
            self.update_params()

            if i % 50 == 0:
                preds = self.predict(self.A[-1])
                acc   = self.utils.accuracy(preds, Y)
                print(f"Iteration {i:>4}  |  Training accuracy: {acc:.4f}")

    #* INFERENCE
    def predict(self, A_out):
        """
        Convert a probability matrix into hard class predictions.
        argmax along axis=0 picks the highest-probability class per example.
        """
        return np.argmax(A_out, axis=0)

    def make_predictions(self, X):
        """Run a forward pass on X and return predicted class indices."""
        self.forward(X)
        return self.predict(self.A[-1])

    def test_prediction(self, index, X, Y):
        """
        Visualize a single example: print predicted vs. true label and display the image.
        """
        image = X[:, index, None]   # (784, 1)
        pred  = self.make_predictions(image)
        label = Y[index]

        print(f"Prediction : {pred[0]}")
        print(f"True label : {label}")

        plt.gray()
        plt.imshow(image.reshape(28, 28), interpolation="nearest")
        plt.title(f"Predicted: {pred[0]}  |  Label: {label}")
        plt.axis("off")
        plt.show()

    def summary(self):
        """Print a compact table of the network architecture."""
        print("=" * 44)
        print(f"  Neural Network  ,  {self.n_layers} layer(s)")
        print("=" * 44)
        print(f"  {'Layer':<12} {'Shape (out x in)':<20} Params")
        print("-" * 44)

        labels = (
            ["Input -> H1"]
            + [f"H{i} -> H{i+1}" for i in range(1, self.n_layers - 1)]
            + ["Hidden -> Out"]
        ) if self.n_layers > 1 else ["Input -> Out"]

        total = 0
        for label, W, b in zip(labels, self.weights, self.biases):
            params = W.size + b.size
            total += params
            print(f"  {label:<12} {str(W.shape):<20} {params:,}")

        print("-" * 44)
        print(f"  {'Total':<12} {'':<20} {total:,}")
        print("=" * 44)

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
 
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    X = mnist.data / 255.0          # shape (70000, 784)
    y = mnist.target.astype(int)    # shape (70000,)

    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # Split
    X_test  = X[:1000].T            # (784, 1000)
    Y_test  = y[:1000]              # (1000,)

    X_train = X[1000:].T            # (784, 60000)
    Y_train = y[1000:]              # (60000,)
    
    nn = NeuralNetwork(
        layer_sizes   = [784, 128, 64, 10],
        learning_rate = 0.1,
        iterations    = 500,
    )

    nn.summary()
    nn.train(X_train, Y_train)
    nn.summary()
    
    # Evaluate on the test set
    test_preds = nn.make_predictions(X_test)
    test_accuracy = Utils.accuracy(test_preds, Y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Visualize one prediction
    nn.test_prediction(index=2, X=X_test, Y=Y_test)