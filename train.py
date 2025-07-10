import sys
from io import open
import os
import numpy as np
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import random
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence

# load data from txt file into padded tensors


def loadData(filename):
    #load data from file
    data = np.loadtxt(filename, dtype=str)
    numbers = data[:, 0]
    labels = data[:, 1].astype(int)
    return numbers, labels

# used to gen labels
def isPalindrome(number):
    if (str(number) == str(number)[::-1]):
        return 1
    else: 
        return 0

# takes a list of str numbers and convert to list of input tensors
def numbersToTensors(numbers):
    # convert numbers to a list of sequences of digits
    number_sequences = [list(map(int, number)) for number in numbers]
    tensorList = []
    # convert to tensor
    for i in range(len(number_sequences)):
        tensor = torch.zeros(len(number_sequences[i]), 1, 10)
        for idx, digit in enumerate(number_sequences[i]):
            tensor[idx, 0, digit] = 1
        tensorList.append(tensor)
    return tensorList

# gen a label tensor for a number
def genLabelTensorFromNumber(number):
    tensor = torch.tensor([isPalindrome(number)], dtype=torch.long)
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        # hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

  
n_hidden = 256  # hidden size of an rnn
n_digits = 10   # input size of an rnn
n_labels = 2    # output size of an rnn

learning_rate = 0.001


# one step of training
def train(model, numberTensor, labelTensor, criterion):
    hidden = model.initHidden()

    model.zero_grad()
    for i in range(numberTensor.size()[0]):
        output, hidden = model(numberTensor[i], hidden)
    
    loss = criterion(output, labelTensor)
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# def randomChoice(l):
#     return l[random.randint(0, len(l) - 1)]

def saveModel(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def main(totalExamples):
    # load and convert data
    trainFilename = f'training_data_{totalExamples}.txt'
    numbersList, labelsList = loadData(trainFilename)
    numberTensorsList = numbersToTensors(numbersList)

    totalExamples = int(totalExamples)
    # n_iters = totalExamples

    rnn = RNN(n_digits, n_hidden, n_labels)

    criterion = nn.NLLLoss()

    n_iters = 100000
    print_every = n_iters / 20
    plot_every = n_iters / 100

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(n_iters):
        # randomly choose an example
        rand_idx = random.randint(0, totalExamples - 1)
        number = numbersList[rand_idx]
        # get label of the example
        label = isPalindrome(number)
        labelTensor = genLabelTensorFromNumber(number)

        output, loss = train(rnn, numberTensorsList[rand_idx], labelTensor, criterion)

        current_loss += loss
        # print training process
        if (iter + 1) % print_every == 0:
            _, top_i = output.topk(1)
            top_i = top_i[0].item()
            correct = 'CORRECT' if top_i == label else 'INCORRECT'
            print('%d %d%% (%s) %.4f %s / %s %s' % ((iter + 1), (iter + 1) / n_iters * 100, timeSince(start), loss, number, top_i, correct))

        if (iter + 1) % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # save trained model
    saveModel(rnn, f'trained_rnn_{totalExamples}_examples.pth')

    # show loss plot
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    

if __name__ == "__main__":
    if (len(sys.argv) != 2) or (sys.argv[1] not in ["200", "2000", "20000"]):
        print("Usage: python train.py <number of training examples>")
        sys.exit(1)

    main(sys.argv[1])


