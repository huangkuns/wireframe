import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torch.nn as nn

def latest(opt):
    latestPath = os.path.join(opt.resume, 'latest.pth.tar')
    if not os.path.exists(latestPath):
        return None
    print('=> Loading the latest checkpoint ' + latestPath)
    return torch.load(latestPath)

def best(opt):
    bestPath = os.path.join(opt.resume, 'best.pth.tar')
    if not os.path.exists(bestPath):
        return None
    print('=> Loading the best checkpoint ' + bestPath)
    return torch.load(bestPath)

def load(opt):
    epoch = opt.epochNum
    if epoch == 0:
        return None
    elif epoch == -1:
        return latest(opt)
    elif epoch == -2:
        return best(opt)
    else:
        modelFile = 'model_' + str(epoch) + '.pth.tar'
        criterionFile = 'criterion_' + str(epoch) + '.pth.tar'
        optimFile = 'optimState_' + str(epoch) +'.pth.tar'
        loaded = {'epoch':epoch, 'modelFile':modelFile, 'criterionFile':criterionFile, 'optimFile':optimFile}
        return loaded

def save(epoch, model, criterion, optimizer, bestModel, loss, opt):
    if isinstance(model, nn.DataParallel):
        model = model.get(0)

    # TODO
    # write model.txt

    modelFile = 'model_' + str(epoch) + '.pth.tar'
    criterionFile = 'criterion_' + str(epoch) + '.pth.tar'
    optimFile = 'optimState_' + str(epoch) +'.pth.tar'

    if bestModel or (epoch % opt.saveEpoch == 0):
        torch.save(model.state_dict(), os.path.join(opt.resume, modelFile))
        torch.save(criterion, os.path.join(opt.resume, criterionFile))
        torch.save(optimizer.state_dict(), os.path.join(opt.resume, optimFile))
        info = {'epoch':epoch, 'modelFile':modelFile, 'criterionFile':criterionFile, 'optimFile':optimFile, 'loss':loss}
        torch.save(info, os.path.join(opt.resume, 'latest.pth.tar'))

    if bestModel:
        info = {'epoch':epoch, 'modelFile':modelFile, 'criterionFile':criterionFile, 'optimFile':optimFile, 'loss':loss}
        torch.save(info, os.path.join(opt.resume, 'best.pth.tar'))
        torch.save(model.state_dict(), os.path.join(opt.resume, 'model_best.pth.tar'))
