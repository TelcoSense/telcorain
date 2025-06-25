#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_telcorain.py
Author: Lukas Kaleta
Date: 2025-05-56
Version: 2.2
Description:
    CNN architecture for the purpose of Telcorain CML precipitation detection.
    Model designed for signle output WD value classification for sample period.
    Inspired by: [1] https://github.com/jpolz/cml_wd_pytorch/tree/main
                 [2] https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py

License:
Contact: 211312@vutbr.cz
"""

""" Imports """
# Import python libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Function definitions """


def init_layer(layer):
    # Initialize a Convolutional layer [2]
    nn.init.xavier_uniform_(layer.weight)


def init_bn(bn):
    # Initialize a Batchnorm layer
    bn.bias.data.zero_()
    bn.weight.data.fill_(1.0)


def output_size(sample_size: int, single_output=True):
    # If True, make CNN output 1 value for whole sample
    # otherwise match output size to sample size
    if single_output:
        return 1
    else:
        return sample_size


""" CNN module architecture """


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, channels_in, channels_out):
        super().__init__()
        self.kernelsize = kernel_size
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(
            self.channels_in, self.channels_out, self.kernelsize, padding="same"
        )  # padding same padds the input data to match the output dimension
        self.conv2 = nn.Conv1d(
            self.channels_out, self.channels_out, self.kernelsize, padding="same"
        )

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)
        return x


class cnn_class(nn.Module):
    def __init__(
        self,
        channels=2,
        sample_size=60,
        kernel_size=3,
        n_fc_neurons=64,
        n_filters=[24, 48, 96, 192],
        single_output=True,
    ):
        super().__init__()
        self.channels = channels
        self.kernelsize = kernel_size
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters
        self.output = output_size(sample_size, single_output)

        ### Convolutional part
        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0])
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1])
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2])
        self.cb4 = ConvBlock(self.kernelsize, self.n_filters[2], self.n_filters[3])
        self.bnc1 = nn.BatchNorm1d(
            self.n_filters[0],
            eps=0,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        init_bn(self.bnc1)

        ### Fully Connected part
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(self.n_filters[3], self.n_fc_neurons)
        self.dense2 = nn.Linear(self.n_fc_neurons, self.n_fc_neurons)
        self.denseOut = nn.Linear(self.n_fc_neurons, self.output)
        self.final_act = nn.Sigmoid()
        self.init_fc()

    def init_fc(self):
        init_layer(self.dense1)
        init_layer(self.dense2)
        init_layer(self.denseOut)

    def forward(self, x):
        ### Conv part
        x = self.cb1(x)
        x = self.bnc1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = torch.mean(x, dim=-1)

        ### FC part
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.final_act(self.denseOut(x))
        return x
