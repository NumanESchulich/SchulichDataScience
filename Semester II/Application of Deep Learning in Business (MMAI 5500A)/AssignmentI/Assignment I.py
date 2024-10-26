import numpy as np

# Load the data
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]


# Dense (Fully connected) layer
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize weights & biases.
        He normal initialization is used to maintain the variance of inputs 
        across layers, preventing vanishing or exploding gradients.
        Biases are initialized to 0.0.
        """
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        # Bias initialization to 0
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        A forward pass through the layer to give z.
        Computing using np.dot + the biases.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dz):
        """
        Backward pass
        """
        # Gradients of weights
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs
        self.dinputs = np.dot(dz, self.weights.T)


# ReLU activation
class ReLu:
    """
    ReLu activation
    """
    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        self.activity = np.maximum(0, z)

    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0


# Softmax activation
class Softmax:
    def forward(self, z):
        """
        Forward pass
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dprobs):
        """
        Backward pass
        """
        # Empty array
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)


# Cross-entropy loss function
class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Clip to prevent division by 0
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # Negative log likelihoods
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)

    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes
        batch_sz, n_class = probs.shape
        # Get the gradient
        self.dprobs = -oh_y_true / probs
        # Normalize the gradient
        self.dprobs = self.dprobs / batch_sz


# Stochastic Gradient Descent (SGD) optimizer
class SGD:
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, learning_rate=1):
        """
        Initialize the optimizer with a learning rate
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


def predictions(probs):
    """
    Convert probabilities to predictions
    """
    return np.argmax(probs, axis=1)


def accuracy(y_preds, y_true):
    """
    Calculate accuracy
    """
    return np.mean(y_preds == y_true)


def one_hot_encode(y_true, n_class):
    """
    One-hot encode labels
    """
    return np.eye(n_class)[y_true]


def forward_pass(X, y_true, oh_y_true):
    """
    A single forward pass through the entire network.
    """
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    dense3.forward(activation2.activity)
    output_activation.forward(dense3.z)
    return output_activation.probs


def backward_pass(probs, y_true, oh_y_true):
    """
    A single backward pass through the entire network.
    """
    crossentropy.backward(probs, oh_y_true)
    output_activation.backward(crossentropy.dprobs)
    dense3.backward(output_activation.dz)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)


# Initialize the network and set hyperparameters
n_epochs = 10
learning_rate = 1
batch_size = 256
n_inputs = 3
n_class = 3
n_neurons1 = 4
n_neurons2 = 8

# Define the layers in the network
dense1 = DenseLayer(n_inputs, n_neurons1)  # First hidden layer
activation1 = ReLu()

dense2 = DenseLayer(n_neurons1, n_neurons2)  # Second hidden layer
activation2 = ReLu()

dense3 = DenseLayer(n_neurons2, n_class)  # Output layer
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(n_epochs):
    print(f'Epoch: {epoch+1}/{n_epochs}')

    epoch_loss = 0
    epoch_accuracy = 0
    n_batches = X_train.shape[0] // batch_size

    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for batch_i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[batch_i:batch_i + batch_size]
        y_batch = y_train_shuffled[batch_i:batch_i + batch_size]

        oh_y_batch = one_hot_encode(y_batch, n_class)
        probs = forward_pass(X_batch, y_batch, oh_y_batch)
        loss = crossentropy.forward(probs, oh_y_batch)
        y_preds = predictions(probs)
        batch_accuracy = accuracy(y_preds, y_batch)

        epoch_loss += loss
        epoch_accuracy += batch_accuracy

        backward_pass(probs, y_batch, oh_y_batch)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)

    epoch_loss /= n_batches
    epoch_accuracy /= n_batches

    test_probs = forward_pass(X_test, y_test, one_hot_encode(y_test, n_class))
    test_loss = crossentropy.forward(test_probs, one_hot_encode(y_test, n_class))
    test_preds = predictions(test_probs)
    test_accuracy = accuracy(test_preds, y_test)

    print(f'Train Loss: {epoch_loss:.4f}\nTrain Accuracy: {epoch_accuracy:.4f}')
    print(f'Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n')
