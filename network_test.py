# from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and: https://github.com/threelittlemonkeys/rnn-encoder-decoder-pytorch/blob/master/model.py
import io
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EncoderRNN(nn.Module):
    def __init__(self, num_input_bins, input_range, hidden_size, rnn_type='gru'):
        super(EncoderRNN, self).__init__()

        assert rnn_type in ['gru']  # TODO: add lstm

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = 1
        self.with_bias = True

        self.num_bins = num_input_bins
        self.bins = np.linspace(input_range[0], input_range[1], self.num_bins)
        # from: https://discuss.pytorch.org/t/convert-tensor-of-floats-to-tensor-of-ordinal-data/109126/3
        self.float_one_hot_encoder = lambda x: nn.functional.one_hot(torch.from_numpy(np.digitize(x, self.bins) - 1),
                                                                     num_classes=self.num_bins)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.num_bins, self.hidden_size, self.num_layers,
                              bidirectional=(self.num_directions == 2))  # batch_first=True ? dropout=True ? bias = True ?
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.num_bins, self.hidden_size, self.num_layers,
                               bidirectional=(self.num_directions == 2))  # batch_first=True ? dropout=True ? bias = True ?

    def forward(self, input, hidden=None):
        print(f'input.shape: {input.shape}')
        embedded = self.float_one_hot_encoder(input).view(1, 1, -1)
        print(f'embedded.shape: {embedded.shape}')

        if hidden is None:
            hidden = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size, device=device)

        output, hidden = self.gru(embedded, hidden)  # note: this is different for lstm

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, num_output_bins, output_range, hidden_size, rnn_type='gru'):  #num_input_bins, input_range, hidden_size, rnn_type='gru'
        super(DecoderRNN, self).__init__()

        assert rnn_type in ['gru']

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = 1
        self.with_bias = True

        self.num_bins = num_output_bins
        self.bins = np.linspace(output_range[0], output_range[1], self.num_bins)
        # from: https://discuss.pytorch.org/t/convert-tensor-of-floats-to-tensor-of-ordinal-data/109126/3
        self.float_one_hot_encoder = lambda x: nn.functional.one_hot(torch.from_numpy(np.digitize(x, self.bins) - 1),
                                                                     num_classes=self.num_bins)

        self.gru = nn.GRU(self.num_bins, self.hidden_size, self.num_layers)

        self.softmax = nn.Softmax(1)

    def forward(self, input, hidden):
        print(f'input.shape: {input.shape}')
        embedded = self.float_one_hot_encoder(input).view(1, 1, -1)
        print(f'embedded.shape: {embedded.shape}')

        # TODO: add attention

        if hidden is None:
            hidden = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size, device=device)

        output, hidden = self.gru(embedded, hidden)  # note: this is different for lstm

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size, device=device)


class EncoderDecoderRNN(nn.Module):
    def __init__(self, num_input_bins, input_range, num_output_bins, output_range, hidden_size, rnn_type='gru'):
        self.encoder = EncoderRNN(num_input_bins, input_range, hidden_size, rnn_type='gru')
        self.decoder = DecoderRNN(num_output_bins, output_range, hidden_size, rnn_type='gru')

    def forward(self, input, hidden=None, decoder_input=None):
        
