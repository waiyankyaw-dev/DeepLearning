from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from cnn_model import CNN

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
NUM_EPOCHS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 1     
OPTIMIZER_DEFAULT = 'ADAM'     
DATA_DIR_DEFAULT = './cifar10_data'

FLAGS = None


def accuracy(predictions, targets):
    if targets.dim() > 1:
        targets = targets.argmax(dim=1)

    preds = predictions.argmax(dim=1)
    correct = (preds == targets).float().mean()
    return correct.item()


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # [B, 10]
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc


def _ensure_flags():
    global FLAGS
    if FLAGS is None:
        class Dummy:
            pass
        FLAGS = Dummy()
        FLAGS.learning_rate = LEARNING_RATE_DEFAULT
        FLAGS.batch_size = BATCH_SIZE_DEFAULT
        FLAGS.num_epochs = NUM_EPOCHS_DEFAULT
        FLAGS.eval_freq = EVAL_FREQ_DEFAULT
        FLAGS.data_dir = DATA_DIR_DEFAULT


def train(batch_size=None, num_epochs=None):
    _ensure_flags()

    if batch_size is None:
        batch_size = FLAGS.batch_size
    if num_epochs is None:
        num_epochs = FLAGS.num_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Batch size:', batch_size)
    print('Num epochs:', num_epochs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=FLAGS.data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=FLAGS.data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )


    model = CNN(n_channels=3, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_DEFAULT.upper() == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=FLAGS.learning_rate,
                              momentum=0.9)

    logs = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        if epoch % FLAGS.eval_freq == 0 or epoch == 1:
            test_loss, test_acc = evaluate(model, test_loader,
                                           criterion, device)
            print(f'Epoch {epoch:3d}/{num_epochs}: '
                  f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                  f'test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')
        else:
            test_loss, test_acc = None, None

        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss)
        logs['train_accuracy'].append(train_acc)
        #store last known test metrics (or None)
        logs['test_loss'].append(test_loss if test_loss is not None else np.nan)
        logs['test_accuracy'].append(test_acc if test_acc is not None else np.nan)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join('checkpoints', 'cnn_cifar10_epochs.pt')
    )
    np.savez(
        os.path.join('checkpoints', 'training_log_epochs.npz'),
        **logs
    )


def main():
    """
    Main function
    """
    _ensure_flags()
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float,
                        default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int,
                        default=NUM_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int,
                        default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int,
                        default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set (in epochs)')
    parser.add_argument('--data_dir', type=str,
                        default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parsed_flags, unparsed = parser.parse_known_args()

    FLAGS = parsed_flags

    main()
