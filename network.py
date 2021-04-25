# from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
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

import matplotlib.pyplot as plt

from build_dataset import DatasetReader


# sort of from: https://discuss.pytorch.org/t/convert-tensor-of-floats-to-tensor-of-ordinal-data/109126/3
# we aren't using it right now, but here is the code
class FloatOneHot:
    def __init__(self, num_bins, data_range, device=torch.device('cpu')):
        self.device = device
        self.num_bins = num_bins
        # from: https://discuss.pytorch.org/t/convert-tensor-of-floats-to-tensor-of-ordinal-data/109126/3
        self.bins = np.linspace(data_range[0], data_range[1], self.num_bins)

    def embed(self, x):
        """
        float one hot embedding
        :param x: 1d numpy array or detached tensor
        :return: returns a 2d tensor
        """
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        return nn.functional.one_hot(torch.from_numpy(np.digitize(x, self.bins) - 1),
                                     num_classes=self.num_bins).type(torch.float).to(self.device)

    def deembed(self, x_emb):
        """
        :param x_emb: 2d numpy array or detached tensor
        :return: np array
        """
        if not isinstance(x_emb, np.ndarray):
            x_emb = x_emb.numpy()

        # maybe use torch.topk(1)
        maxes = np.argmax(x_emb, axis=1)

        return self.bins[maxes]


class Encoder(nn.Module):
    def __init__(self, input_range, mult_factor, embedding_dim, hidden_size, rnn_type='gru'):  # or should we have the caller make the rnn layer
        super(Encoder, self).__init__()

        assert rnn_type in ['gru']  # TODO: add lstm

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size

        # TODO: should we make as inputs
        self.num_layers = 1
        self.num_directions = 1

        self.mult_factor = mult_factor
        self.embedding_dim = embedding_dim
        self.input_range = input_range

        self.embedding = nn.Embedding(int((input_range[1] - input_range[0])*mult_factor), embedding_dim)

        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, bidirectional=(self.num_directions == 2))  # batch_first=True ? dropout=True ? bias = True ?

    def forward(self, input, hidden=None):
        """

        :param input: 1d tensor of sequential data (len seq_len)
        :param hidden: optional
        :return: output (shape (seq_len, 1, hidden_size)), output_hidden
        """
        # because the embedding layer expects integer values we want to rescale the input data
        rescaled_input = self.to_rescaled_input(input)
        # the rnn expects input with the dimensions (seq_len, batch, input_size).
        # we will only be doing one batch at a time.
        embedded = self.embedding(rescaled_input).view(input.shape[0], 1, self.embedding_dim)

        if hidden is None:
            # TODO: add lstm, not the hidden state is different for lstm than it is for gru
            hidden = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size)  # TODO: device=device

        output, hidden = self.rnn(embedded, hidden)  # note: this is different for lstm

        return output, hidden

    def to_rescaled_input(self, x):
        """
        rescales the input data that is in the range `self.input_range` to be integers.
        Basically just just multiplies every value by `self.mult_factor`.
        :param x: tensor of floats
        :return: tensor of longs
        """
        return torch.clamp(((x - self.input_range[0]) * self.mult_factor).type(torch.long),
                           0, int((self.input_range[1]-self.input_range[0])*self.mult_factor-1))


class Decoder(nn.Module):
    def __init__(self, output_range, mult_factor, embedding_dim, hidden_size, rnn_type='gru'):  #num_input_bins, input_range, hidden_size, rnn_type='gru'
        super(Decoder, self).__init__()

        assert rnn_type in ['gru']

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = 1

        self.num_embedding = int((output_range[1] - output_range[0])*mult_factor)
        self.mult_factor = mult_factor
        self.embedding_dim = embedding_dim
        self.output_range = output_range

        self.embedding = nn.Embedding(self.num_embedding, embedding_dim)
        self.emb_relu = nn.ReLU()

        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, bidirectional=(self.num_directions == 2))  # batch_first=True ? dropout=True ? bias = True ?

        self.dense = nn.Linear(self.hidden_size*self.num_directions, self.num_embedding, True)
        self.softmax = nn.LogSoftmax(dim=2)  # note, this should NOT be dim=1 like they did in the seq2seq_translation_tutorial

    def forward(self, input, hidden):
        """

        :param input: 1d tensor of sequential data (len seq_len)
        :param hidden: optional
        :return: output tensor (shape (seq_len, 1, hidden_size)), output_hidden
        """
        # TODO: add attention

        if len(input.shape) == 0:
            length=1
        else:
            length=len(input)
        rescaled_input = self.to_rescaled_input(input)
        embedded = self.embedding(rescaled_input).view(length, 1, self.embedding_dim)
        embedded = self.emb_relu(embedded)

        if hidden is None:
            hidden = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size)  # todo: device=device

        output, hidden = self.rnn(embedded, hidden)  # note: this is different for lstm

        output_linear = self.dense(output)

        output_encodded = self.softmax(output_linear)

        return output_encodded, hidden

    def network_output_tensors_to_numbers(self, y_tensor):
        """
        Note, this will detach the tensor so can't be use with back propagate

        :param y_tensor: 1d or 2d tensor
        :return: 1d tensor
        """
        topv, topi = y_tensor.topk(1)
        output = topi.squeeze().reshape(-1).detach()/self.mult_factor + self.output_range[0]  # detach from history as input

        return output

    def to_rescaled_input(self, x):
        """
        rescales the input data that is in the range `self.input_range` to be integers.
        Basically just just multiplies every value by `self.mult_factor`.
        :param x: tensor of floats
        :return: tensor of longs
        """
        return torch.clamp(((x - self.output_range[0]) * self.mult_factor).type(torch.long),
                           0, int((self.output_range[1]-self.output_range[0])*self.mult_factor-1))


class EncoderDecoder:
    def __init__(self, input_range, input_mult_factor, input_embedding_dim, output_range, output_mult_factor,
                 output_embedding_dim, hidden_size,tfr=0.5, rnn_type='gru'):
        super(EncoderDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.encoder = Encoder(input_range, input_mult_factor, input_embedding_dim, hidden_size, rnn_type=rnn_type)
        self.decoder = Decoder(output_range, output_mult_factor, output_embedding_dim, hidden_size, rnn_type=rnn_type)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.NLLLoss()
        self.teacher_forcing_ratio = tfr

    def run_idk(self, input_tensor, output_length, hidden=None):
        """
        runs the encoder_decoder with the input, `input_tensor`, getting `output_length` points
        :param input_tensor: 1d tensor
        :param output_length:
        :param hidden: optional
        :return: 1d tensor of predicted values
        """
        encoder_outputs, hidden = self.encoder(input_tensor, hidden)  # if you get an error use torch.from_numpy

        decoder_input = torch.tensor([0])  # TODO: or should it be a best guess for first value, idk

        output = torch.zeros(output_length)

        for di in range(output_length):
            # note, self.decoder will rescale the data and so it expects the non-rescaled values
            # it returns the outputs in the rescaled form
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            output_val = self.decoder.network_output_tensors_to_numbers(decoder_output)
            output[di] = output_val.detach()[0]
            decoder_input = output_val

        return output

    def train(self, input_tensor, target_tensor):
        target_rescaled_tensor = self.decoder.to_rescaled_input(target_tensor).reshape(-1,1)

        encoder_hidden = self.encoder.forward(input_tensor)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs, hidden = self.encoder(input_tensor)

        loss = 0

        decoder_input = torch.tensor([0])

        decoder_hidden = encoder_hidden

        use_tf = True if random.random() < self.teacher_forcing_ratio else False

        if use_tf:
            for di in range(target_length):
                # note, self.decoder will rescale the data and so it expects the non-rescaled values
                # it returns the outputs in the rescaled form
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                output_val = self.decoder.network_output_tensors_to_numbers(decoder_output)
                loss += self.criterion(decoder_output[:,0,:],target_rescaled_tensor[di])
                decoder_input = target_tensor[di]      
        else:
            for di in range(target_length):
                # note, self.decoder will rescale the data and so it expects the non-rescaled values
                # it returns the outputs in the rescaled form
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                output_val = self.decoder.network_output_tensors_to_numbers(decoder_output)
                loss += self.criterion(decoder_output[:,0,:],target_rescaled_tensor[di])
                decoder_input = output_val

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length


if __name__ == '__main__':
    test_net = EncoderDecoder((0,30), 10, 20, (0,1000), 0.5, 10, 1000)

    train_loss = []

    dsr = DatasetReader('dataset', 'Boulder Creek', '663', '06730200', 40, (1960, 2016))
    for x, y in dsr:
        this_loss = test_net.train(torch.from_numpy(x),torch.from_numpy(y))
        train_loss.append(this_loss)
        print(this_loss)

    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.show()

    dsr = DatasetReader('dataset', 'Boulder Creek', '663', '06730200', 2, (2013, 2050))
    for x, y in dsr:
        y_hat = test_net.run_idk(torch.from_numpy(x), y.shape[0])
        print(y, y_hat.numpy())
        plt.plot(np.arange(y.shape[0]), y, label='y')
        plt.plot(np.arange(y_hat.shape[0]), y_hat.numpy(), label='y_hat')
        plt.show()
