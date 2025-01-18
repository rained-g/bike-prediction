import numpy as np
from torch import nn


class Module_LSTM(nn.Module):
    def __init__(self, hidden_size=512, output_size=96, input_size=10, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        input_seq, _ = self.lstm(input_seq)
        input_seq = input_seq[:, -1, :]
        input_seq = self.linear(input_seq)
        input_seq = self.relu(input_seq)
        return input_seq

if __name__ == '__main__':

    model = Module_LSTM(input_size=14, output_size=512,num_layers=3, hidden_size=256)

    sum = 0
    for i, module in model.named_parameters():
        print(i, module.numel())
        sum += module.numel()
    print(sum)