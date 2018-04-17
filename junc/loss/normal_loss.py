
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import math
import scipy.stats as stats

pixel_mean = Variable(torch.FloatTensor(
    [115.9839754, 126.63120922, 137.73309306]).view(1, 3, 1, 1))
eps_div = 1e-20


class Loss(nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        H = opt.hype
        self.batch_size = opt.batch_size
        self.num_bin = H['num_bin']
        self.grid_h  = self.grid_w = H['grid_size']
        self.num_grids = self.grid_h * self.grid_w
        # self.max_ratio = 8.

        self.w0 = H['loss_weights'][:2]
        self.w1 = H['loss_weights'][2:4]

    def forward(self, junction_logits, junction_loc, bin_logits, bin_residual, junc_conf_var, junc_res_var, bin_conf_var, bin_res_var):
        junction_logits = junction_logits.view(-1, 2, self.grid_h, self.grid_w)
        bin_logits = bin_logits.view(-1, 2,
                                     self.num_bin, self.grid_h, self.grid_w)

        # junc_conf_result = F.softmax(junction_logits)
        # junc_conf_result = junc_conf_result[:, 1, :, :]
        # junc_res_result = junction_loc

        # bin_conf_result = F.softmax(bin_logits)
        # bin_conf_result = bin_conf_result[:, 1, :, :, :]
        # bin_res_result = bin_residual

        junc_conf_loss = torch.nn.NLLLoss(weight=None, reduce=False)(
            F.log_softmax(junction_logits, dim=1), junc_conf_var)
        junc_res_loss = torch.nn.MSELoss(
            reduce=False)(junction_loc, junc_res_var)
        # original
        #junc_conf_loss = torch.mean(junc_conf_loss)
        #junc_res_loss = torch.mean(junc_res_loss)
        
        junc_conf_loss = torch.mean(junc_conf_loss)
        mask_junc = junc_conf_var == 1
        junc_res_loss = torch.mul(junc_res_loss, mask_junc.float())
        junc_res_loss = torch.sum(junc_res_loss) / (torch.sum(mask_junc) + 1e-20)

        
        # N x 2 x H x W
        bin_conf_loss = torch.nn.NLLLoss(weight=None, reduce=False)(F.log_softmax(
            bin_logits.view(-1, 2, self.grid_h * self.num_bin, self.grid_w), dim=1), bin_conf_var.view(-1, self.grid_h * self.num_bin, self.grid_w))
        bin_conf_loss = bin_conf_loss.view(-1,
                                           self.num_bin, self.grid_h, self.grid_w)
        # N x self.num_bin x H x W
        bin_res_loss = torch.nn.MSELoss(
            reduce=False)(bin_residual, bin_res_var)
        # N x self.num_bin x H x W

        mask_bin = junc_conf_var.unsqueeze(1).float()
        bin_conf_loss = torch.mul(bin_conf_loss, mask_bin)

        bin_res_loss = torch.mul(bin_res_loss, mask_bin)
        bin_conf_loss = torch.sum(bin_conf_loss)
        bin_res_loss = torch.sum(bin_res_loss)

        bin_conf_loss = bin_conf_loss / \
            float(torch.sum(mask_bin) * self.num_bin + eps_div)
        bin_res_loss = bin_res_loss / \
            float(torch.sum(mask_bin) * self.num_bin + eps_div)

        loss = junc_conf_loss * self.w0[0] + junc_res_loss * self.w0[1]
        loss += bin_conf_loss * self.w1[0] + bin_res_loss * self.w1[1]

        return (loss,
                junc_conf_loss, junc_res_loss,
                bin_conf_loss, bin_res_loss
                )

