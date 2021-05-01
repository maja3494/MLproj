# from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import io
import unicodedata
import string
import re
import random
import numpy as np
from time import time
from random import shuffle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.functional import normalize

import matplotlib.pyplot as plt

from build_dataset import DatasetReader

from scipy.ndimage import gaussian_filter1d


class Net(nn.Module):
    def __init__(self, lr, device):
        super().__init__()
        # class attributes
        self.device = device
        # network layers
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=50, padding=0, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(
            in_channels=50, out_channels=10, padding=0, kernel_size=2, stride=1)
        self.conv3 = nn.Conv1d(
            in_channels=10, out_channels=10, padding=0, kernel_size=2, stride=1)
        self.trans_conv1 = torch.nn.ConvTranspose1d(
            in_channels=10, out_channels=20, padding=0, kernel_size=2, stride=1)
        self.trans_conv2 = torch.nn.ConvTranspose1d(
            in_channels=20, out_channels=50, padding=0, kernel_size=2, stride=1)
        self.trans_conv3 = torch.nn.ConvTranspose1d(
            in_channels=50, out_channels=1, padding=0, kernel_size=3, stride=1)
        self.lin1 = nn.Linear(167, 82)
        self.lin2 = nn.Linear(82, 10)
        self.lin3 = nn.Linear(10, 50)
        self.lin4 = nn.Linear(50, 87)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lin_drop = nn.Dropout(p=0.6)
        # loss function
        self.loss_fn = nn.L1Loss(reduction='sum')
        self.loss_param = nn.Parameter(torch.tensor(
            data=11, dtype=torch.float32, device=self.device, requires_grad=True))
        # auxilliary fields
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.5)
        # self.optimizer = optim.AdamW(self.parameters())
        self.train_losses = []
        self.test_losses = []
        # self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr/1000)

    def forward(self, x):
        # conv1
        x = x.view(-1, 1, 173)
        x = self.tanh(x)
        # x = self.sig(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # print('shape1:', x.shape)
        x = self.lin_drop(x)
        x = self.lin1(x)
        x = self.tanh(x)
        x = self.lin_drop(x)
        x = self.lin2(x)
        x = self.tanh(x)
        x = self.lin_drop(x)
        x = self.lin3(x)
        x = self.tanh(x)
        x = self.lin_drop(x)
        x = self.lin4(x)
        x = self.tanh(x)
        x = self.lin_drop(x)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        # print('shape2:', x.shape)
        return x[:][0][:]  # remove channel dimension

    def train_step(self, x_train, y_train):
        self.train()
        yhat = self(x_train)
        yhat = normalize(yhat[0], dim=0, p=1)
        y_train = normalize(y_train[0], dim=0, p=1)
        # self.loss_fn(input=yhat, target=y_train)
        # loss = -((torch.dot(yhat, y_train) + self.loss_param)**2)
        loss = -((torch.dot(yhat, y_train) + 11)**2)
        # print('loss:', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss.item()

    def evaluate(self, x_test, y_test):
        with torch.no_grad():
            self.eval()
            yhat = self(x_test)
            loss = self.loss_fn(input=yhat, target=y_test)
            return loss.item()

    def train_loop(self, num_epochs, train_loader, test_loader):
        self.train_losses = []
        self.test_losses = []

        start = time()

        for epoch in range(num_epochs):
            print('epoch:', epoch)
            epoch_train_loss = 0
            epoch_test_loss = 0

            # evaluating
            for x_batch, y_batch in test_loader:
                # send data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # calculate loss
                test_loss = self.evaluate(x_batch, y_batch)
                epoch_test_loss += test_loss

            # training
            for x_batch, y_batch in train_loader:
                # send data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # calculate loss
                train_loss = self.train_step(
                    x_batch, y_batch)
                epoch_train_loss += train_loss

            self.train_losses.append(epoch_train_loss)
            self.test_losses.append(epoch_test_loss)

        end = time()
        train_time = end - start

        return train_time, self.train_losses, self.test_losses


def getmin(dsr):
    '''
    return the minimum lengths of x and y data
    '''
    dsr_iter = iter(dsr)
    min_x = 1000000
    min_y = 1000000
    for x, y in dsr_iter:
        x_len = len(x)
        if x_len < min_x:
            min_x = x_len
        y_len = len(y)
        if y_len < min_y:
            min_y = y_len
    return min_x, min_y


def rand_resize(dsr, min_x, min_y):
    '''
    randomly downsample data to resize it
    return resized data so that all x data are same length and all y data are same length
    '''
    x_data = []
    y_data = []
    dsr_iter = iter(dsr)
    for x, y in dsr_iter:
        if (min_x > len(x)):
            print('uh oh x')
        if (min_y > len(y)):
            print('uh oh y')
        x_enum = list(enumerate(x))
        y_enum = list(enumerate(y))
        shuffle(x_enum)
        shuffle(y_enum)
        x_shrunk = x_enum[:min_x]
        y_shrunk = y_enum[:min_y]
        x_reordered = sorted(x_shrunk)
        y_reordered = sorted(y_shrunk)
        x_denum = np.array(x_reordered)[:, 1]
        y_denum = np.array(y_reordered)[:, 1]
        x_data.append(x_denum.tolist())
        y_data.append(y_denum.tolist())

    return x_data, y_data


def split_data(x_data, y_data):
    dataset = TensorDataset(x_data, y_data)
    num_examples = len(dataset)
    train_len = int(0.8*num_examples)  # 80|20 split
    test_len = num_examples - train_len
    train_dataset, test_dataset = random_split(
        dataset=dataset, lengths=[train_len, test_len])
    return train_dataset, test_dataset


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100
    print('device:', device)

    model = Net(lr=10, device=device).to(device)

    dsr = DatasetReader('dataset', 'Ark', '369',
                        '07094500', 1, (1980, 2021))
    x_min_len, y_min_len = getmin(dsr)
    # hardcoded values because dsr has incosistent output
    x_min_len, y_min_len = 173, 91
    # print('x_min_len:', x_min_len)
    # print('y_min_len:', y_min_len)
    x_data, y_data = rand_resize(dsr, x_min_len, y_min_len)
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)
    train_dataset, test_dataset = split_data(x_data, y_data)

    # data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # load model parameters
    # model = torch.load('./best4.pth')

    # training loop
    print('training...')
    train_time, train_losses, test_losses = model.train_loop(
        num_epochs=num_epochs, train_loader=train_loader, test_loader=test_loader)
    print('finished training')
    print('training time (seconds):', train_time)

    # save model
    torch.save(model, './last.pth')

    print('loss_param:', model.state_dict()['loss_param'])

    # plot loss
    plt.figure(1)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['training loss', 'testing loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    fig = plt.figure()
    test_iter = iter(test_loader)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        x, y = next(test_iter)
        x = x.to(device)
        yhat = model(x)
        y = y.cpu().numpy()
        yhat = yhat.cpu().detach().numpy()
        yhat = yhat[0]
        y = y[0]
        plt.plot(range(len(y)), y, label='y')
        plt.plot(range(len(yhat)), yhat, label='yhat')
        plt.legend()
    plt.tight_layout()
    plt.show()
