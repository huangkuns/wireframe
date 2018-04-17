import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable

import numpy as np
import time
import pdb
import importlib

from model.networks.inception_v2 import inception_v2


class inception(nn.Module):
    def __init__(self, classes, opt):
        super(inception, self).__init__()
        self.classes = classes
        self.n_classes = 2
    
        decoder_module = importlib.import_module('model.networks.{}_decoder'.format(opt.decoder))

        self.decoder_ = decoder_module.DecodeNet(opt, 'train')
        self.base_net = inception_v2(num_classes = 2, with_bn=opt.hype.get('batchnorm', True))
        
    def forward(self, im_data, junc_conf, junc_res, bin_conf, bin_res):
        # junc_conf, junc_res, bin_conf, bin_res
        """ input includes:
            im_data
            gt_junctions
            junc_conf
            junc_residual
            bin_conf
            bin_residual
	    """

        batch_size = im_data.size(0)
        base_feat = self.base_net(im_data)
        preds = self.decoder_(base_feat)
        return preds
