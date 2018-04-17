import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import importlib

def setup(opt, checkpoint, model):
    criterion = None
    criterionHandler = importlib.import_module('models.' + opt.netType + '-criterion')

    if checkpoint != None:
        criterionPath = os.path.join(opt.resume, checkpoint['criterionFile'])
        assert os.path.exists(criterionPath), '=> WARNING: Saved criterion not found: ' + criterionPath
        print('=> Resuming criterion from ' + criterionPath)
        criterion = torch.load(criterionPath)
        criterionHandler.initCriterion(criterion, model)
    else:
        print('=> Creating criterion from file: models/' + opt.netType + '-criterion.py')
        criterion = criterionHandler.createCriterion(opt, model)

    if opt.GPU:
        criterion = criterion.cuda()

    return criterion
