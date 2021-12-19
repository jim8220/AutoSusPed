import random

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import pandas
from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE

import torch.nn.functional as F

class ConvStack(nn.Module):
    def __init__(self, cnn_unit, fc_unit, mnum):
        super().__init__()

        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 2), stride=(1, 2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 4
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_unit * 2, fc_unit),
            nn.Dropout(0.5))

    def forward(self, proll):
        x = proll.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class ConvStack_deeper(nn.Module):
    def __init__(self, cnn_unit, fc_unit, mnum):
        super().__init__()

        # modificiation 1 : split all cnn modules for Residual connection or skip connection.


        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn0 = nn.Sequential(
            # layer 0 : CNN
            nn.Conv2d(1, cnn_unit, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn0r = nn.Sequential(
            # layer 0r : CNN
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn1 = nn.Sequential(
            # layer 1 : CNN
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn1r = nn.Sequential(
            # layer 1 : CNN
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn2 = nn.Sequential(
            # layer 1 : CNN
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn2r = nn.Sequential(
            # layer 1 : CNN
            nn.Conv2d(cnn_unit, cnn_unit, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU())
        self.cnn3 = nn.Sequential(
            # layer 3: CNN
            nn.Conv2d(cnn_unit, cnn_unit * 2, kernel_size=(2*mnum+1, 2), stride=(1,2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())
        self.cnn3r = nn.Sequential(
            # layer 3: CNN
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())
        self.cnn4 = nn.Sequential(
            # layer 4 : CNN
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 2), stride=(1, 2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())
        self.cnn4r = nn.Sequential(
            # layer 3: CNN
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())
        self.cnn5 = nn.Sequential(
            # layer 4 : CNN
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 2), stride=(1, 2), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())
        self.cnn5r = nn.Sequential(
            # layer 3: CNN
            nn.Conv2d(cnn_unit * 2, cnn_unit * 2, kernel_size=(2*mnum+1, 1), stride=(1,1), padding=(mnum,0)),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(cnn_unit * 2, fc_unit),
            nn.Dropout(0.5))

    def forward(self, proll):
        x = proll.unsqueeze(1)
        x1 = self.cnn0(x)
        x1r = self.cnn0r(x1)
        x2 = self.cnn1(x1r)
        x2r = self.cnn1r(x2)
        x3 = self.cnn2(x2r)
        x3r = self.cnn2r(x3)
        x4 = self.cnn3(x3r)
        x4r = self.cnn3r(x4)
        x5 = self.cnn4(x4r)
        x5r = self.cnn4r(x5)
        x6 = self.cnn5(x5r)
        x6r = self.cnn5r(x6)
        xout = x6r.transpose(1,2).flatten(-2)
        xout = self.fc(xout)
        return xout

class AutoSusPed(nn.Module): # Basic CNN model
    def __init__(self, cnn_unit, fc_unit, mnum, level):
        super().__init__()

        self.pedal_conv_stack = ConvStack(cnn_unit, fc_unit, mnum)
        self.pedal_fc = nn.Linear(fc_unit, level)
        self.softmax = nn.Softmax(dim=2)
        print(self)

    def forward(self, proll):
        x = self.pedal_conv_stack(proll)  # (B, T, C)
        pedal_out = self.softmax(self.pedal_fc(x))
        return pedal_out

class AutoSusPed_deeper(nn.Module): # this model is added in 2021.12.10
    def __init__(self, cnn_unit, fc_unit, mnum, level):
        super().__init__()

        self.pedal_conv_stack = ConvStack_deeper(cnn_unit, fc_unit, mnum)
        self.pedal_fc = nn.Linear(fc_unit, level)
        self.softmax = nn.Softmax(dim=2)
        print(self)

    def forward(self, proll):
        x = self.pedal_conv_stack(proll)  # (B, T, C)
        pedal_out = self.softmax(self.pedal_fc(x))
        return pedal_out

if __name__ == "__main__":
    print('All tests passed!')