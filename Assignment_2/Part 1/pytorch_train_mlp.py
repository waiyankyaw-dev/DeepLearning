from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.

    Args:
        predictions: 2D tensor of size [num_samples, n_classes] with model
                     outputs (logits).
        targets: 1D tensor of size [num_samples] with integer class labels.

    Returns:
        accuracy: scalar float, the accuracy of predictions in percent.
    """
    with torch.no_grad():  #Temporarily disables gradient tracking since we’re only evaluating. Saves memory & speed
        predicted_classes = predictions.argmax(dim=1)
        correct = (predicted_classes == targets).float().mean().item()
    return correct * 100.0

"""Example:
predicted_classes = tensor([0, 1, 1])

targets = tensor([0, 0, 1])

comparison → [True, False, True].

.float() converts True→1.0, False→0.0: [1.0, 0.0, 1.0].

.mean() averages: (1 + 0 + 1) / 3 = 0.666….

.item() converts tensor scalar to Python float 0.666….

Multiply by 100 → 66.666… %."""

def train(X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Performs training and evaluation of MLP model.

    If X_train / y_train are not provided, it creates a make_moons dataset
    (1000 samples, noise=0.2) and splits it into 80% train / 20% test,
    exactly like your NumPy training code.

    NOTE: The model is evaluated on the whole test set every eval_freq epochs.
    """
    if FLAGS is not None:
        dnn_hidden_units = FLAGS.dnn_hidden_units
        learning_rate = FLAGS.learning_rate
        max_steps = FLAGS.max_steps
        eval_freq = FLAGS.eval_freq
    else:
        dnn_hidden_units = DNN_HIDDEN_UNITS_DEFAULT
        learning_rate = LEARNING_RATE_DEFAULT
        max_steps = MAX_EPOCHS_DEFAULT
        eval_freq = EVAL_FREQ_DEFAULT

    #turn "20,30,40" into [20, 30, 40]
    hidden_units = []
    if dnn_hidden_units is not None and dnn_hidden_units != '':
        hidden_units = [int(h) for h in dnn_hidden_units.split(',')]

    np.random.seed(42)
    torch.manual_seed(42)

    if X_train is None or y_train is None:
        print("Generating new moon dataset...")
        X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        print("Using provided dataset...")
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

    n_inputs = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_tensor = torch.from_numpy(X_train).float().to(device) #.float() makes features float32
    y_train_tensor = torch.from_numpy(y_train).long().to(device) #.long() makes labels int64
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    model = MLP(n_inputs=n_inputs, n_hidden=hidden_units,
                n_classes=n_classes).to(device)

    # CrossEntropyLoss = log-softmax + negative log-likelihood
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Starting training...")
    print(f"Hidden units: {hidden_units if hidden_units else '[] (no hidden layer)'}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps (epochs): {max_steps}")
    print(f"Eval frequency: {eval_freq}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Using device: {device}")

    for step in range(max_steps):
        model.train()

        #forward pass on the whole training set
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)

        # backward pass
        optimizer.zero_grad() #clears old gradients stored in model parameters (otherwise they accumulate).
        loss.backward()  #Auto-diff: computes gradients of loss w.r.t. every parameter
        optimizer.step()  #Updates each parameter: param = param - lr * param.grad

        if (step % eval_freq == 0) or (step == max_steps - 1):
            model.eval()  #Switches to evaluation mode (turns off dropout, etc.). Again, good habit
            with torch.no_grad():  #No gradients during evaluation
                train_logits = model(X_train_tensor) #Get logits on entire training set
                test_logits = model(X_test_tensor)

                train_loss = criterion(train_logits, y_train_tensor).item() #.item() to convert from tensor to Python float
                test_loss = criterion(test_logits, y_test_tensor).item()

                train_acc = accuracy(train_logits, y_train_tensor)
                test_acc = accuracy(test_logits, y_test_tensor)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)

            print(
                f"Step {step:4d}: "
                f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
                f"Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%"
            )

    print("Training complete!")

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str,
                        default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float,
                        default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int,
                        default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int,
                        default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    FLAGS, unparsed = parser.parse_known_args()
    main()
