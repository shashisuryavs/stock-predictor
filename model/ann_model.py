import numpy as np
import pandas as pd

class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)  
        self.bias_hidden = np.zeros((1, self.hidden_size))  

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)  
        self.bias_output = np.zeros((1, self.output_size))  

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)

        return self.output

    def backward(self, X, y, learning_rate=0.001):
        # Compute error at the output layer
        output_error = y - self.output  
        output_delta = output_error * self.sigmoid_derivative(self.output)  

        # Reshape output_delta if needed
        output_delta = output_delta.reshape(-1, 1)

        # Backpropagate to Hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)  
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)  

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 

    def sigmoid_derivative(self, x):
        return x * (1 - x)  

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def accuracy(self, y_true, y_pred):
        """Calculates the accuracy by comparing rounded predictions to actual values."""
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = y_true.shape[0]
        return (correct_predictions / total_predictions) * 100
