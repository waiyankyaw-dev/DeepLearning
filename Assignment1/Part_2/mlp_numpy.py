from modules import * 

class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes the multi-layer perceptron object.
        
        Args:
            n_inputs (int): Number of inputs (i.e., dimension of an input vector).
            n_hidden (list of int): List of integers, where each integer is the number of units in each hidden layer.
            n_classes (int): Number of classes of the classification problem (i.e., output dimension of the network).
        """
        self.layers = []
        
        prev_units = n_inputs
        for hidden_units in n_hidden:
            self.layers.append(Linear(prev_units, hidden_units))
            self.layers.append(ReLU())
            prev_units = hidden_units
        
        self.layers.append(Linear(prev_units, n_classes))
        self.layers.append(SoftMax())
        
    def forward(self, x):
        """
        Predicts the network output from the input by passing it through several layers.
        
        Args:
            x (numpy.ndarray): Input to the network.
            
        Returns:
            numpy.ndarray: Output of the network.
        """
        out = x  
        
        for layer in self.layers:
            out = layer.forward(out)
        
        return out

    def backward(self, dout):
        """
        Performs the backward propagation pass given the loss gradients.
        
        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent.
        """
        for layer in self.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                layer.params['weight'] -= learning_rate * layer.grads['weight']
                layer.params['bias'] -= learning_rate * layer.grads['bias']