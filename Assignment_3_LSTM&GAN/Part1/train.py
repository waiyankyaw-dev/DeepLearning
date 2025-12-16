from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, device, config):
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Data to device and correct types
        batch_inputs = batch_inputs.float().to(device)
        batch_targets = batch_targets.long().to(device)

        # Ensure shape is (Batch, Seq, Dim). 
        # If input_dim=1, dataset gives (Batch, Seq, 1), so no change needed usually.
        # But if dataset gives (Batch, Seq), we unsqueeze.
        if config.input_dim == 1 and batch_inputs.dim() == 2:
            batch_inputs = batch_inputs.unsqueeze(-1)

        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = criterion(logits, batch_targets)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        acc = accuracy(logits, batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        if step % 50 == 0:
            print(f'Train Step [{step}/{len(data_loader)}] Loss: {losses.val:.4f} Acc: {accuracies.val:.3f}')

    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.float().to(device)
        batch_targets = batch_targets.long().to(device)

        if config.input_dim == 1 and batch_inputs.dim() == 2:
            batch_inputs = batch_inputs.unsqueeze(-1)

        logits = model(batch_inputs)
        loss = criterion(logits, batch_targets)
        acc = accuracy(logits, batch_targets)

        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

    return losses.avg, accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset setup
    # Note: dataset returns (Seq, Dim)
    full_dataset = PalindromeDataset(
        input_length=config.input_length,
        total_len=config.data_size,
        one_hot=(config.input_dim == 10) # Auto-detect one-hot based on input dim
    )

    train_size = int(config.portion_train * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = LSTM(
        seq_length=config.input_length,
        input_dim=config.input_dim,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    # Scheduler helps lower LR as we get closer to solution
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(config.max_epoch):
        print(f"\nEpoch {epoch+1}/{config.max_epoch}")
        train_loss, train_acc = train(model, train_dloader, optimizer, criterion, device, config)
        val_loss, val_acc = evaluate(model, val_dloader, criterion, device, config)
        
        print(f"Validation Accuracy: {val_acc:.3f} | Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step()

        if val_acc >= 0.999:
            print("Converged!")
            return val_acc

    return val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--input_length', type=int, default=19, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=20, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int, default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8, help='Portion of the total dataset used for training')

    config = parser.parse_args()
    main(config)