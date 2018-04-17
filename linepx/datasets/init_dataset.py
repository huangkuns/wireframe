import os
import torch
import importlib
import subprocess
import math
import numpy as np
import random
import ref


def create(opt, split):
    info = exec(opt, ref.cacheFile)
    dataset = importlib.import_module('datasets.' + opt.dataset)

    return dataset.getInstance(info, opt, split)


def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data

    print("=> Generating list of data")


    if opt.testOnly:
        dataFile = ref.data_root / 'v1.1' /  'test.txt'
    else:
        dataFile = ref.data_root / 'v1.1' /  'train.txt'

    with open(dataFile, 'r') as f:
        dataList = f.read().splitlines()
    dataList = [x[:-4] for x in dataList]

    random.shuffle(dataList)
    # set train/val 4800/200

    trainImg = [opt.data / "{}_rgb.png".format(x) for x in dataList]
    trainLine = [opt.data / "{}_line.png".format(x) for x in dataList]

    if opt.testOnly:
        val_list = dataList[:]
        train_list = dataList[:]

        valImg = trainImg[:]
        valLine = trainLine[:]
        trainImg = trainImg[:]
        trainLine = trainLine[:]
    else:
        val_list = dataList[:200]
        train_list = dataList[200:]

        valImg = trainImg[:200]
        valLine = trainLine[:200]
        trainImg = trainImg[200:]
        trainLine = trainLine[200:]

    numTrain = len(trainImg)
    numVal = len(valImg)
    print('#Training images: {}'.format(numTrain))
    print('#Val images: {}'.format(numVal))

    info = {'basedir': opt.data,
            'train': {
                'imagePath'  : trainImg,
                'linePath'   : trainLine
                },
            'val': {
                'imagePath'  : valImg,
                'linePath'   : valLine
                }
            }

    torch.save(info, cacheFile)

    return info
