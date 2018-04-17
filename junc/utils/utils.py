from numpy.random import randn
import ref
import torch
import numpy as np

def adjust_learning_rate(optimizer, epoch, LR, LR_param):
    #lr = LR * (0.1 ** (epoch // dropLR))
    LR_policy = LR_param.get('lr_policy', 'step')
    if LR_policy == 'step':
        steppoints = LR_param.get('steppoints', [4, 7, 9, 10])
        lrs = LR_param.get('lrs', [0.01, 0.001, 0.0001, 0.00001, 0.000001])
        assert len(lrs) == len(steppoints) + 1
        
        lr = None
        for idx, steppoint in enumerate(steppoints):
            if epoch > steppoint:
                continue
            elif epoch <= steppoint:
                lr = lrs[idx]
                break
        if lr is None:
            lr = lrs[-1]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
  for e in ref.shuffleRef:
    x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
  return x
