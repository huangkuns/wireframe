import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import math
import scipy.stats as stats


from torch.autograd import Variable
pixel_mean = Variable(torch.FloatTensor(
    [115.9839754, 126.63120922, 137.73309306]).view(1, 3, 1, 1))
eps_div = 1e-20


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


class Decoder(nn.Module):
    def __init__(self, input_dim, channel, out1, out2):
        super(Decoder, self).__init__()
        self.nInput = input_dim
        self.conv1 = BasicConv2d(self.nInput, channel, 3, padding=1)
        self.conv1_1 = nn.Conv2d(channel, out1[0], 1, padding=0)
        self.conv1_2 = nn.Conv2d(channel, out1[1], 1, padding=0)

        self.conv2 = BasicConv2d(self.nInput, channel, 3, padding=1)
        self.conv2_1 = nn.Conv2d(channel, out2[0], 1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, out2[1], 1, padding=0)

    def forward(self, input):
        x0 = self.conv1(input)
        junction_logits = self.conv1_1(x0)
        junction_loc = self.conv1_2(x0)

        x1 = self.conv2(input)
        bin_logits = self.conv2_1(x1)
        bin_residual = self.conv2_2(x1)

        return junction_logits, junction_loc, bin_logits, bin_residual


class DecodeNet(nn.Module):
    def __init__(self, opt, phase):
        super(DecodeNet, self).__init__()
        H = opt.hype
        self.batch_size = opt.batch_size
        self.num_bin = H['num_bin']
        self.grid_h = self.grid_w = H['grid_size']
        self.num_grids = self.grid_h * self.grid_w
        self.out_size = self.grid_h * self.grid_w * self.batch_size
        if opt.balance:
            out1 = (3 * H['max_len'], 2 * H['max_len'])
            out2 = (2 * H['num_bin'] * H['max_len'], H['num_bin'] * H['max_len'])
        else:
            out1 = (2 * H['max_len'], 2 * H['max_len'])
            out2 = (2 * H['num_bin'] * H['max_len'], H['num_bin'] * H['max_len'])
        
        decodeFeats = H.get('decodeFeats', 256) # 256 is the reported structure in paper.
        self.decoder = Decoder(decodeFeats, 256, out1, out2)

    def forward(self, input):
        (junction_logits,
         junction_loc,
         bin_logits,
         bin_residual
         ) = self.decoder(input)
        return (
            junction_logits,
            junction_loc,
            bin_logits,
            bin_residual
        )
