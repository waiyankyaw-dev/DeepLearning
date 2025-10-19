import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs (features) for each data point.
        - max_epochs: Maximum number of training cycles (epochs) to perform.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias) as a zero vector.
        """
        self.n_inputs = n_inputs  
        self.max_epochs = max_epochs  
        self.learning_rate = learning_rate  
        self.weights = np.zeros(n_inputs + 1)  
        
    def forward(self, input_vec):
        """
        Predicts label from input using the current weights.
        Args:
            input_vec (np.ndarray): Input array of training data, shape (N, n_inputs) where N is the number of samples.
        Returns:
            int: Predicted labels (1 or -1) for all samples as a numpy array.
        """
        N = input_vec.shape[0]  
        X_with_bias = np.c_[input_vec, np.ones((N, 1))]  
        z = np.dot(X_with_bias, self.weights)  
        return np.where(z >= 0, 1., -1.) 
        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron using the standard batch gradient descent algorithm.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points, converted to (N, n_inputs).
            labels (np.ndarray): Array of expected output values (-1 or 1) corresponding to training points.
        """
        training_inputs = np.array(training_inputs)  
        N = training_inputs.shape[0]  
        X_with_bias = np.c_[training_inputs, np.ones((N, 1))] 
        for epoch in range(self.max_epochs): 
            preds = self.forward(training_inputs)  
            # print(f"Epoch {epoch + 1}: predictions = {preds}")
            misclassified = (preds * labels < 0)  
            if not np.any(misclassified):  
                print(f"Converged at epoch {epoch}")  
                break  
            mis_X = X_with_bias[misclassified]  
            mis_y = labels[misclassified]  
            num_mis = len(mis_y)  
            # print(f"number of missclassified points at epoch {epoch}: {num_mis}")
            gradient = - (1 / num_mis) * np.sum(mis_y[:, np.newaxis] * mis_X, axis=0)
            self.weights -= self.learning_rate * gradient  
            # print(f"Epoch {epoch + 1}: weights = {self.weights}")
        else:  
            print("Reached max(100) epochs")  

def generate_dataset(mean_pos, mean_neg, cov_scale=0.5):
    cov = [[cov_scale, 0], [0, cov_scale]]  

    pos_points = np.random.multivariate_normal(mean_pos, cov, 100)  
    neg_points = np.random.multivariate_normal(mean_neg, cov, 100)  

    pos_labels = np.ones(100) 
    neg_labels = -np.ones(100)  

    all_points = np.vstack((pos_points, neg_points))  
    all_labels = np.hstack((pos_labels, neg_labels))  

    indices = np.random.permutation(200)
    all_points = all_points[indices]
    all_labels = all_labels[indices]

    train_points = all_points[:160]  
    train_labels = all_labels[:160]
    test_points = all_points[160:]  
    test_labels = all_labels[160:]

    return train_points, train_labels, test_points, test_labels

def plot_dataset_and_boundary(train_points, train_labels, test_points, test_labels, weights, title):
    plt.figure(figsize=(8, 6))
    # Plot training points
    plt.scatter(train_points[train_labels == 1, 0], train_points[train_labels == 1, 1], c='blue', label='Train +1', alpha=0.6)
    plt.scatter(train_points[train_labels == -1, 0], train_points[train_labels == -1, 1], c='red', label='Train -1', alpha=0.6)
    # Plot test points
    plt.scatter(test_points[test_labels == 1, 0], test_points[test_labels == 1, 1], c='blue', marker='x', label='Test +1', s=50)
    plt.scatter(test_points[test_labels == -1, 0], test_points[test_labels == -1, 1], c='red', marker='x', label='Test -1', s=50)
    # Plot decision boundary if weights learned
    if np.all(weights != 0):
        w1, w2, b = weights
        if w2 != 0:
            x1 = np.linspace(-5, 5, 100)
            x2 = - (w1 * x1 + b) / w2
            plt.plot(x1, x2, 'g--', label='Decision Boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    testcases = [
        ([2, 2], [-2, -2], 0.5, "Default: Separable"),
        ([2, 2], [2.5, 2.5], 0.5, "Close Means"),
        ([2, 2], [-2, -2], 8.0, "High Variance"),
        ([2, 2], [2.5, 2.5], 8.0, "Close Means + High Variance")
    ]
    
    for mean_pos, mean_neg, cov_scale, title in testcases:
        print(f"\n--- {title} ---")
        train_points, train_labels, test_points, test_labels = generate_dataset(mean_pos, mean_neg, cov_scale)
        
        perceptron = Perceptron(n_inputs=2)  
        perceptron.train(train_points, train_labels)
        
        test_preds = perceptron.forward(test_points)
        accuracy = np.mean(test_preds == test_labels)
        print(f"Test accuracy: {accuracy:.4f}")
        
        plot_dataset_and_boundary(train_points, train_labels, test_points, test_labels, perceptron.weights, title)