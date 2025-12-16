from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        """
        seq_length : number of input time steps (T-1 in the assignment)
        input_dim  : dimensionality of x^(t)  (here 1 – a single digit)
        hidden_dim : dimensionality of h^(t)
        output_dim : number of classes (10 digits)
        batch_size : minibatch size (not strictly needed but kept for clarity)
        """
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        # h^(t) = tanh(W_hx x^(t) + W_hh h^(t-1) + b_h)
        # W_hx: maps input -> hidden, no bias (bias lives in W_hh)
        self.W_hx = nn.Linear(input_dim, hidden_dim, bias=False)
        # W_hh: maps previous hidden -> hidden, with bias b_h
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # o^(t) = W_ph h^(t) + b_o
        self.W_ph = nn.Linear(hidden_dim, output_dim, bias=True)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_hx.weight)
        nn.init.xavier_uniform_(self.W_hh.weight)
        nn.init.xavier_uniform_(self.W_ph.weight)

        if self.W_hh.bias is not None:
            nn.init.zeros_(self.W_hh.bias)  #biases start at zero
        if self.W_ph.bias is not None:
            nn.init.zeros_(self.W_ph.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device

        #h^(0) = 0 
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        #Loop over time: t = 0 ... seq_len-1
        for t in range(seq_len):
            x_t = x[:, t, :] # (batch_size, input_dim)
            # h^(t) = tanh(W_hx x^(t) + W_hh h^(t-1) + b_h)
            h_t = torch.tanh(self.W_hx(x_t) + self.W_hh(h_t))

        #only the last hidden state h^(T-1)
        #o^(T) = W_ph h^(T) + b_o
        o_t = self.W_ph(h_t)  # (batch_size, output_dim)

        return o_t
