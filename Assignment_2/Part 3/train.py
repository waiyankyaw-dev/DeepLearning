from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN


def train(config, return_history=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = VanillaRNN(
        seq_length=config.input_length,
        input_dim=config.input_dim,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
        batch_size=config.batch_size
    ).to(device)

    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    history = {
        "step": [],
        "loss": [],
        "accuracy": []
    }

    model.train()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):


        batch_inputs = batch_inputs.float().to(device)
        batch_targets = batch_targets.long().to(device)

        batch_inputs = batch_inputs.unsqueeze(-1)

        logits = model(batch_inputs)                

        loss = criterion(logits, batch_targets)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        predictions = torch.argmax(logits, dim=1)   # (batch_size,)
        correct = (predictions == batch_targets).float().sum().item()
        accuracy = correct / batch_targets.size(0)

        if return_history:
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["accuracy"].append(accuracy)

        if step % 10 == 0:
            print('Step {:5d} | Loss {:6.4f} | Accuracy {:5.3f}'.format(
                step, loss.item(), accuracy))

        if step >= config.train_steps:
            break

    print('Done training.')

    if return_history:
        return model, history
    else:
        return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    train(config, return_history=False)
