import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initCriterion(criterion, model):
	pass

def createCriterion(opt, model):
    criterion = nn.MSELoss()

    return criterion
