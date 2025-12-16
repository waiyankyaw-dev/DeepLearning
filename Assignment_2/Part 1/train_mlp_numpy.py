import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from mlp_numpy import MLP  
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.

    Args:
    predictions: 2D float array of size [number_of_data_samples, n_classes]
    targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding

    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """

    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(pred_classes == true_classes) * 100

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size='full', X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        batch_size: 'full' for batch gradient descent, or integer for mini-batch size
        X_train, y_train: Pre-generated training data (optional)
        X_test, y_test: Pre-generated test data (optional)
    """
    # TODO: Load your data here
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    if X_train is None or y_train is None:
        print("Generating new moon dataset...")
        X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
        
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42
        )
    else:
        print("Using provided dataset...")
    
    hidden_units = [int(units) for units in dnn_hidden_units.split(',')]

    # Infer input dimension and number of classes from the data
    n_inputs = X_train.shape[1]          # e.g. 2 for all these toy datasets
    n_classes = y_train.shape[1]         # number of columns in one-hot labels
    
    model = MLP(n_inputs= n_inputs, n_hidden=hidden_units, n_classes= n_classes)
    criterion = CrossEntropy()


    
    n_train = X_train.shape[0]
    if batch_size == 'full':
        actual_batch_size = n_train
        n_batches = 1
        print("Using Batch Gradient Descent (full dataset)")
    else:
        # Mini-batch or Stochastic Gradient Descent
        actual_batch_size = int(batch_size)
        n_batches = int(np.ceil(n_train / actual_batch_size))
        if actual_batch_size == 1:
            print("Using Stochastic Gradient Descent (batch_size=1)")
        else:
            print(f"Using Mini-batch Gradient Descent (batch_size={actual_batch_size})")
    
    print(f"Training samples: {n_train}, Batches per epoch: {n_batches}")
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Starting training...")
    for step in range(max_steps):
        #shuffling data each epoch for mini-batch SGD
        if batch_size != 'full':
            indices = np.random.permutation(n_train)
            X_epoch = X_train[indices]
            y_epoch = y_train[indices]
        else:
            X_epoch = X_train
            y_epoch = y_train
        
        epoch_loss = 0
        
        # Process in batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * actual_batch_size
            end_idx = min((batch_idx + 1) * actual_batch_size, n_train)
            
            X_batch = X_epoch[start_idx:end_idx]
            y_batch = y_epoch[start_idx:end_idx]
            
            # Forward pass
            train_output = model.forward(X_batch)
            batch_loss = criterion.forward(train_output, y_batch)
            epoch_loss += batch_loss * (end_idx - start_idx)
            
            # Backward pass
            dout = criterion.backward()
            model.backward(dout)
            
            # Update parameters
            model.update_parameters(learning_rate)
        
        # average loss over epoch
        epoch_loss /= n_train
        
        if step % eval_freq == 0 or step == max_steps - 1:
            # evaluate on test set
            test_output = model.forward(X_test)
            test_loss = criterion.forward(test_output, y_test)
            
            # calculate accuracies on full sets
            full_train_output = model.forward(X_train)
            train_acc = accuracy(full_train_output, y_train)
            test_acc = accuracy(test_output, y_test)
            
            train_losses.append(epoch_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Step {step:4d}: Train Loss = {epoch_loss:.4f}, Test Loss = {test_loss:.4f}, "
                  f"Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")
    
    print("Training complete!")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'model': model,
        'batch_size': batch_size,
        'steps': list(range(0, max_steps, eval_freq)) + [max_steps-1],
        'X_train': X_train,  
        'y_train': y_train,  
        'X_test': X_test,    
        'y_test': y_test     
    }

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=str, default='full',
                        help='Batch size: "full" for batch GD, or integer for mini-batch')
    FLAGS = parser.parse_args()
    
    results = train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, 
                   FLAGS.max_steps, FLAGS.eval_freq, FLAGS.batch_size)
    return results

if __name__ == '__main__':
    main()