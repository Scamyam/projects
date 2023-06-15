import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.biases = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]

    def forward(self, X):
        activation = X
        activations = [activation]
        zs = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        return activations, zs

    def backward(self, X, y, activations, zs, learning_rate):
        m = X.shape[1]
        deltas = [None] * self.num_layers
        deltas[-1] = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(zs[-1])

        for i in range(self.num_layers - 2, 0, -1):
            deltas[i] = np.dot(self.weights[i].T, deltas[i+1]) * self.sigmoid_derivative(zs[i-1])

        for i in range(self.num_layers - 2, -1, -1):
            self.weights[i] -= learning_rate * np.dot(deltas[i+1], activations[i].T) / m
            self.biases[i] -= learning_rate * np.mean(deltas[i+1], axis=1, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            activations, zs = self.forward(X)
            self.backward(X, y, activations, zs, learning_rate)

            if (epoch + 1) % 100 == 0:
                loss = self.mean_squared_error(activations[-1], y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def mean_squared_error(self, y_pred, y_true):
        return np.mean(np.power(y_pred - y_true, 2))


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # Output

layer_sizes = [2, 4, 3, 1]  # Number of neurons in each layer (including input and output layers)

# Create a neural network
nn = NeuralNetwork(layer_sizes)

# Train the neural network
nn.train(X.T, y.T, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X.T)
print("Predictions:")
print(predictions.T)
