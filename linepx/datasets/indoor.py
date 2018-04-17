from torch.utils.data.dataset import *
import torch
import os
import numpy as np
import cv2
import datasets.transforms as t
import pickle

class indoorDist(Dataset):
    def __init__(self, imageInfo, opt, split):
        self.imageInfo = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']

    def __getitem__(self, index):
        imgPath = self.imageInfo['imagePath'][index]
        imgPath = str(imgPath)
        image = cv2.imread(imgPath)
        image = image / 255.
        image = self.preprocess(image)

        linePath = self.imageInfo['linePath'][index]
        linePath = str(linePath)
        line = cv2.imread(linePath, 0)
        line = self.preprocessLine(line)

        image = torch.from_numpy(image).float()
        line = torch.from_numpy(line).float()

        # if self.opt.testOnly:
        #     imgName = imgPath.split('/')[-1].replace('_rgb.png', '')
        #     return image, line, imgName
        # else:
        #     return image, line
        imgName = imgPath.split('/')[-1].replace('_rgb.png', '')
        return image, line, imgName

    def __len__(self):
        return len(self.imageInfo['imagePath'])

    def preprocess(self, im):
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        im = np.asarray(im)
        im = t.normalize(im, mean, std)
        im = np.transpose(im, (2, 0, 1))
        return im

    def preprocessLine(self, line):
        line = np.asarray(line)
        tmp = np.zeros((1, self.opt.imgDim, self.opt.imgDim))
        tmp[0, :, :] = line
        line = tmp
        return line

    def postprocess(self):
        def process(im):
            mean = torch.Tensor([0.485, 0.456, 0.406])
            std = torch.Tensor([0.229, 0.224, 0.225])
            im = np.transpose(im, (1, 2, 0))
            im = t.unNormalize(im, mean, std)
            return im
        return process

    def postprocessLine(self):
        def process(im):
            im = np.transpose(im, (1, 2, 0))
            return im
        return process

def getInstance(info, opt, split):
    myInstance = indoorDist(info, opt, split)
    return myInstance
