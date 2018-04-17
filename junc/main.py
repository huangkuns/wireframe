import os
import time
import datetime
import cv2

# suppress warning.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

import _init_paths
from opts import opts

import ref
from utils.utils import adjust_learning_rate

from datasets.utils import create_dataset
from datasets.junction import *

from model.inception import inception

from pathlib import Path
import importlib

project_root = ref.root_dir
debug_dataset = False

def init_trainer(trainer_name, opt, model, criterion, optimizer):
    trainerLib = importlib.import_module('trainer.{}_trainer'.format(trainer_name))
    return trainerLib.Trainer(opt, model, criterion, optimizer)

def init_criterion(criterion_name, opt):
    criterionLib = importlib.import_module('loss.{}_loss'.format(criterion_name))
    return criterionLib.Loss(opt)

def init_optimizer(opt, model):
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                         lr=opt.lr, 
                                         momentum=0.9, 
                                         weight_decay=1e-6
                                         )
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=opt.lr, betas=(opt.momentum, 0.999), 
                                    eps=1e-8,
                                    weight_decay=opt.weightDecay
                                    )
    elif opt.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                             alpha=0.9, 
                                             lr=opt.lr, 
                                             eps=1e-6, 
                                             weight_decay=1e-6
                                             )
    return optimizer

# init
def init_dataset(H, split, size_info=True):
    if split == 'train':
        return Junction(H, split='train')
    elif split == 'val':
        return Junction(H, split='val')
    elif split == 'test':
        return JunctionTest(H, split='test', size_info=size_info)
    else:
        raise NotImplementedError

def init_dataloader(opt):
    test_loader = torch.utils.data.DataLoader(
        init_dataset(opt.hype, 'test'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1
    ) if opt.test else None

    train_loader = torch.utils.data.DataLoader(
        init_dataset(opt.hype, opt.split),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=max(4, opt.batch_size),
        pin_memory=True
    ) if not opt.test else None
    
    if train_loader is not None:
        print("length of train_loader: {}".format(len(train_loader)))

    return train_loader, test_loader

def check_dataset(opt, split):
    ## options for creating.
    H = opt.hype
    if opt.create_dataset:
        create_dataset(H, split, use_mp=True); 
        print("finished creating dataset.")
        return True
    return False

def init_folder(opt):
    if not os.path.isdir(opt.saveDir): 
        os.mkdir(opt.saveDir)

def main():
    opt = opts().parse();  H = opt.hype
    if check_dataset(opt, 'train'):
        return
    init_folder(opt)
    
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(opt.gpu)
    
    lr_param = {}
    lr_param['lr_policy'] = 'step'
    lr_param['steppoints'] = H['steppoints'] #[9, 12, 15] #[8, 10, 12], [20, 24, 26]
    opt.lr_param = lr_param
    net = opt.net; opt.cuda = True # use cuda by default

    # build model/cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if net == 'inception':
        model = inception(['-', '+'], opt)
    elif net == 'resnet':
        raise NotImplementedError

    # use cuda by default.
    if opt.cuda: model.cuda()
    
    ## Init
    optimizer = init_optimizer(opt, model)
    criterion = init_criterion(opt.criterion, opt)
    trainer = init_trainer(opt.trainer, opt, model, criterion, optimizer)#Trainer(model, opt)
    train_loader, test_loader = init_dataloader(opt)

    ## Train
    if not opt.test:
        trainer.train(train_loader, val_loader=None)
    else:
        epoch_test = opt.checkepoch
        checkpoint = ref.output_root / "{expID}/model_{epoch_test}.pth".format(expID=opt.exp, epoch_test=epoch_test)   
        trainer.test(test_loader, epoch_test, checkpoint)
    
if __name__ == '__main__':
    main()
