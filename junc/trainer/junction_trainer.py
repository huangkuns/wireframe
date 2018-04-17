import os
import cv2

import time
from tqdm import tqdm

import numpy as np
import scipy.io as sio

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


from utils.plotter import Plotter
from utils.utils import AverageMeter
from progress.bar import Bar
from model.networks.generate_cell_anchors import generate_anchors, generate_bin_anchors
from pathlib import Path

import pickle
import ref
import datetime

eps_div = 1e-7


def imwrite(imgname, img):
    # in case imgname is a Path obj.
    cv2.imwrite(str(imgname), img)


class Trainer():
    def __init__(self, opt, model, criterion, optimizer, optimState=None):
        self.model = model
        self.optimState = optimState
        self.opt = opt
        H = opt.hype
        
        self.optimizer = optimizer
        self.criterion = criterion

        if self.optimState is not None:
            self.optimizer.load_state_dict(self.optimState)
            
        self.num_bin = self.opt.hype['num_bin']
        self.grid_h = self.grid_w = self.opt.hype['grid_size']
        self.batch_size = opt.batch_size
        self.num_grids = self.grid_h * self.grid_w
        self.max_ratio = 1000.

        self.w0 = H['loss_weights'][:2]
        self.w1 = H['loss_weights'][2:4]

        self.plot_ = Plotter(opt.hype)
        self.saveDir = Path(self.opt.saveDir)

        if opt.lr_param['lr_policy'] == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=opt.lr_param['steppoints'], gamma=0.1)

        self.pretrained_model = ''

        self.test_junc_thresh_list = [0.1 * x for x in range(10)]
        self.train_junc_thresh_list = [0.1, 0.5]

        self.log_interval = 100
        self.logfile = ref.logdir / "{}.log".format(self.opt.exp)
        with open(self.logfile, 'a') as fn:
            fn.write('\n{}\n'.format(datetime.datetime.now()))

    def train(self, train_loader, val_loader=None):
        for epoch in range(1, self.opt.max_epochs + 1):
            self.lr_scheduler.step()
            self.step(epoch, train_loader)
            torch.save(self.model, self.saveDir / 'model_{}.pth'.format(epoch))
            if val_loader is not None:
               self.step(epoch, val_loader, is_val=True, split='val')

    def test(self, dataLoader, epoch, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
            
        test_res_dir = ref.result_dir / self.opt.exp / str(epoch)
        self.test_dir = test_res_dir
        if not self.test_dir.is_dir():
            os.makedirs(self.test_dir)

        iter_per_epoch = len(dataLoader)
        bar = Bar('==>', max=iter_per_epoch * len(self.test_junc_thresh_list))
        
        start = time.time()
        for i, input_data in enumerate(dataLoader):
            inp, imgname, size_info = input_data

            input_var = torch.autograd.Variable(inp).float().cuda()
            junc_conf_var = None # torch.autograd.Variable(junc_conf).long().cuda()
            junc_res_var = None # torch.autograd.Variable(junc_res).float().cuda()
            bin_conf_var = None # torch.autograd.Variable(bin_conf).long().cuda()
            bin_res_var = None # torch.autograd.Variable(bin_res).float().cuda()

            (junction_logits,
                junction_loc,
                bin_logits,
                bin_residual
             ) = self.model(input_var, junc_conf_var, junc_res_var, bin_conf_var, bin_res_var)

            if self.plot_ is not None:
                junc_conf_result, junc_res_result, bin_conf_result, bin_res_result = self.conf_loc(
                        junction_logits, junction_loc, bin_logits, bin_residual)

                num_inp = inp.size(0)
                split='test'

                for junc_thresh in self.test_junc_thresh_list:
                    bar.suffix = '{split} epoch: [{0}][{1}/{2}]| threshold: {3}'.format(
                        epoch, i, iter_per_epoch, junc_thresh, split=split)
                    
                    test_thresh_dir = self.test_dir / str(int(10 * junc_thresh))
                    if not os.path.isdir(test_thresh_dir):
                        os.mkdir(test_thresh_dir)

                    for idx_inside in range(num_inp):
                        filename = imgname[idx_inside]
                        ## theta_thresh = 0.5
                        tmp2 = {}
                        tmp2['h'], tmp2['w'] = size_info[idx_inside]
                        (image, tmp2['junctions'], tmp2['thetas'], tmp2['theta_confs']) = self.plot_.plot_junction_simple(
                            inp[idx_inside].numpy(),
                            [junc_conf_result.data.cpu()[idx_inside].numpy(),
                             junc_res_result.data.cpu()[idx_inside].numpy(),
                             bin_conf_result.data.cpu()[idx_inside].numpy(),
                             bin_res_result.data.cpu()[idx_inside].numpy()
                             ],
                            junc_thresh=junc_thresh, 
                            theta_thresh=0.5, 
                            size_info = size_info[idx_inside],
                            keep_conf=True
                        )

                        imwrite("{}/{}_5.png".format(test_thresh_dir, filename), image)
                        sio.savemat("{}/{}.mat".format(test_thresh_dir, filename), tmp2)
                        with open(test_thresh_dir / "{}_5.pkl".format(filename), 'wb') as fn:
                            pickle.dump(tmp2, fn)
                    bar.next()
        bar.finish()
        return 0


    def step(self, epoch, dataLoader, is_training=True, is_val=False, is_testing=False, split='train'):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        iter_per_epoch = len(dataLoader)
        bar = Bar('==>', max=iter_per_epoch * len(self.test_junc_thresh_list) if is_testing else iter_per_epoch)
        
        Loss, LossJuncConf, LossJuncRes, LossBinConf, LossBinRes = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        start = time.time()
        for i, input_data in enumerate(dataLoader):
            inp, junc_conf, junc_res, bin_conf, bin_res, imgname, size_info = input_data

            input_var = torch.autograd.Variable(inp).float().cuda()
            junc_conf_var = torch.autograd.Variable(junc_conf).long().cuda()
            junc_res_var = torch.autograd.Variable(junc_res).float().cuda()
            bin_conf_var = torch.autograd.Variable(bin_conf).long().cuda()
            bin_res_var = torch.autograd.Variable(bin_res).float().cuda()

            (junction_logits,
                junction_loc,
                bin_logits,
                bin_residual
             ) = self.model(input_var, junc_conf, junc_res, bin_conf, bin_res)

            if is_training:
                (loss,
                    junc_conf_loss,
                    junc_res_loss,
                    bin_conf_loss,
                    bin_res_loss
                 ) = self.criterion(junction_logits, junction_loc, bin_logits, bin_residual, junc_conf_var, junc_res_var, bin_conf_var, bin_res_var)

                if not is_val:
                    self.optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    self.optimizer.step()
                Loss.update(
                    loss.data[0],
                    inp.size(0)
                )
                LossJuncConf.update(
                    junc_conf_loss.data[0],
                    inp.size(0)
                )
                LossBinConf.update(
                    bin_conf_loss.data[0],
                    inp.size(0)
                )
                LossJuncRes.update(
                    junc_res_loss.data[0],
                    inp.size(0)
                )
                LossBinRes.update(
                    bin_res_loss.data[0],
                    inp.size(0)
                )

                split = 'val' if is_val else 'train'
                elapsed  = time.time() - start
                remaining_seconds = elapsed * (iter_per_epoch - i) / (i+1)
                mins, secs = divmod(remaining_seconds, 60)
                hr, mins = divmod(mins, 60)
                rsecs = "{:02d}:{:02d}:{:02d}".format(int(hr), int(mins), int(secs))
                logStr = '{split} epoch: [{0}][{1}/{2}]| LR: {3:.6f}| Speed: {4:.2f}/s| Remaining: {5} | Loss {Loss.avg:.6f}|  JuncConf {JC.avg:.6f}| JuncRes {JR.avg:.6f}| BinConf {BC.avg:.6f}| BinRes {BR.avg:.6f}'.format(epoch, i, iter_per_epoch, self.lr_scheduler.get_lr()[0], float(i+1.)/elapsed, rsecs, split=split, Loss=Loss, JC=LossJuncConf, JR=LossJuncRes, BC=LossBinConf, BR=LossBinRes)
                if i == iter_per_epoch - 1:
                    with open(self.logfile, 'a') as fn:
                        fn.write('{}\n'.format(logStr))
                bar.suffix = logStr
                bar.next()
        bar.finish()
        return 0

    def step_balance(self, epoch, dataLoader, is_training=True, split='train'):
        self.model.train()

        valres_dir = Path('valres/{}/{}'.format('train' if is_training else 'val', self.opt.exp))
        if not valres_dir.is_dir():
            os.mkdir(valres_dir)

        iter_per_epoch = len(dataLoader)
        bar = Bar('==>', max=iter_per_epoch * len(self.test_junc_thresh_list) if is_testing else iter_per_epoch)
        
        Loss, LossJuncConf, LossJuncRes, LossBinConf, LossBinRes = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        start = time.time()
        for i, input_data in enumerate(dataLoader):
            inp, junc_conf, junc_res, bin_conf, bin_res, imgname, size_info, mask = input_data

            input_var = torch.autograd.Variable(inp).float().cuda()
            junc_conf_var = torch.autograd.Variable(junc_conf).long().cuda()
            junc_res_var = torch.autograd.Variable(junc_res).float().cuda()
            bin_conf_var = torch.autograd.Variable(bin_conf).long().cuda()
            bin_res_var = torch.autograd.Variable(bin_res).float().cuda()

            (junction_logits,
                junction_loc,
                bin_logits,
                bin_residual
             ) = self.model(input_var, junc_conf, junc_res, bin_conf, bin_res)

            (loss,
                junc_conf_loss,
                junc_res_loss,
                bin_conf_loss,
                bin_res_loss
             ) = self.criterion(junction_logits, junction_loc, bin_logits, bin_residual, junc_conf_var, junc_res_var, bin_conf_var, bin_res_var)

            if not is_val:
                self.optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                self.optimizer.step()
            Loss.update(
                loss.data[0],
                inp.size(0)
            )
            LossJuncConf.update(
                junc_conf_loss.data[0],
                inp.size(0)
            )
            LossBinConf.update(
                bin_conf_loss.data[0],
                inp.size(0)
            )
            LossJuncRes.update(
                junc_res_loss.data[0],
                inp.size(0)
            )
            LossBinRes.update(
                bin_res_loss.data[0],
                inp.size(0)
            )

            split = 'val' if is_val else 'train'
            elapsed  = time.time() - start
            remaining_seconds = elapsed * (iter_per_epoch - i) / (i + 1)
            mins, secs = divmod(remaining_seconds, 60)
            hr, mins = divmod(mins, 60)
            rsecs = "{:02d}:{:02d}:{:02d}".format(int(hr), int(mins), int(secs))

            bar.suffix = '{split} epoch: [{0}][{1}/{2}]| LR: {3:.6f}| Speed: {4:.2f}/s| Remaining: {5} | Loss {Loss.avg:.6f}|  JuncConf {JC.avg:.6f}| JuncRes {JR.avg:.6f}| BinConf {BC.avg:.6f}| BinRes {BR.avg:.6f}'.format(
                epoch, i, iter_per_epoch, self.lr_scheduler.get_lr()[0], float(i+1.)/elapsed, rsecs, split=split, Loss = Loss, JC=LossJuncConf, JR=LossJuncRes, BC=LossBinConf, BR=LossBinRes)
            bar.next()

        bar.finish()
        return 0
    
    def resume(self, model_path):
        pretrained_model = torch.load(model_path)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def conf_loc(self, junction_logits, junction_loc, bin_logits, bin_residual):
        junc_conf_result = F.softmax(junction_logits, dim=1)
        junc_conf_result = junc_conf_result[:, 1, :, :]
        junc_res_result = junction_loc

        bin_logits = bin_logits.view(-1, 2, self.num_bin, self.grid_h, self.grid_w)
        bin_conf_result = F.softmax(bin_logits, dim=1)
        bin_conf_result = bin_conf_result[:, 1, :, :, :]
        bin_res_result = bin_residual

        return junc_conf_result, junc_res_result, bin_conf_result, bin_res_result
