from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        # Input Gate (i)
        self.W_ix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_ih = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Forget Gate (f)
        self.W_fx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_fh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output Gate (o)
        self.W_ox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_oh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Cell Candidate (g)
        self.W_gx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_gh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Prediction (p)
        self.W_ph = nn.Linear(hidden_dim, output_dim, bias=True)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Xavier initialization for weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)

        # CRITICAL FIX: Initialize Forget Gate bias to 1.0
        # This prevents gradients from vanishing at the start of training
        init.constant_(self.W_fh.bias, 1.0)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_len, _ = x.size()
        device = x.device

        # h^(0) and c^(0) initialized to zeros
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        # Loop over time steps
        for t in range(seq_len):
            x_t = x[:, t, :] # (batch_size, input_dim)

            # Equations
            g_t = self.tanh(self.W_gx(x_t) + self.W_gh(h_t))
            i_t = self.sigmoid(self.W_ix(x_t) + self.W_ih(h_t))
            f_t = self.sigmoid(self.W_fx(x_t) + self.W_fh(h_t))
            o_t = self.sigmoid(self.W_ox(x_t) + self.W_oh(h_t))

            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t

        # Prediction based on last hidden state
        p_t = self.W_ph(h_t)
        return p_t