from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.

        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of
                      units in each hidden layer.
            n_classes: number of classes of the classification problem
                       (i.e., output dimension of the network).
        """
        super(MLP, self).__init__()

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]

        layers = []
        prev_units = n_inputs

        for hidden_units in n_hidden:
            layers.append(nn.Linear(prev_units, hidden_units))
            layers.append(nn.ReLU())
            prev_units = hidden_units

        #output layer (no activation here – CrossEntropyLoss will apply
        #softmax internally to these logits)
        layers.append(nn.Linear(prev_units, n_classes))

        #pack everything into a single Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.

        Args:
            x: input to the network, shape [batch_size, n_inputs]

        Returns:
            out: raw class scores (logits), shape [batch_size, n_classes]
        """
        out = self.network(x)
        return out
