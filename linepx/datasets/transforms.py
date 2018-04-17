import math
import numpy as np
import torch
import random

# ipt is nparray with dimension (height, width, channel)
# xml is nparray with dimension (height, width)

def addNoise(ipt, miu, std):
    noise = np.random.normal(miu, std, ipt.shape)
    noise = np.float32(noise)
    return ipt + noise


def thAddNoise(ipt, miu, std):
    noise = np.random.normal(miu, std, ipt.size())
    noise = torch.from_numpy(np.float32(noise))
    return ipt + noise

def scaleRGB(ipt):
    return np.float32(ipt/255)

def unScaleRGB(ipt):
    opt = ipt*255
    opt = opt.astype(np.uint8)
    return opt

def normalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] - mean[0]) / std[0]
    ipt[:][:][1] = (ipt[:][:][1] - mean[1]) / std[1]
    ipt[:][:][2] = (ipt[:][:][2] - mean[2]) / std[2]
    return ipt

def unNormalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] * std[0]) + mean[0]
    ipt[:][:][1] = (ipt[:][:][1] * std[1]) + mean[1]
    ipt[:][:][2] = (ipt[:][:][2] * std[2]) + mean[2]
    return ipt

def randomFlip(ipt, xml):
    if random.uniform(0, 1) > 0.5:
        ipt = np.fliplr(ipt).copy()
        xml = np.fliplr(xml).copy()
    return ipt, xml

def randomCrop(ipt, xml, size):
    origH = ipt.shape[0]
    origW = ipt.shape[1]
    newH = size[0]
    newW = size[1]
    startH = random.randint(0, origH - newH)
    startW = random.randint(0, origW - newW)
    ipt = ipt[startH : startH+newH, startW : startW+newW, :]
    xml = xml[startH : startH+newH, startW : startW+newW]
    return ipt, xml

def randomSizeCrop(ipt, xml, LowBound):
    newH = math.floor(random.uniform(LowBound, 1)*ipt.shape[0])
    while newH%8 != 0:
        newH -= 1
    newW = math.floor(random.uniform(LowBound, 1)*ipt.shape[1])
    while newW%8 != 0:
        newW -= 1
    return randomCrop(ipt, xml, (newH, newW))
