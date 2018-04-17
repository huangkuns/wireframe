import torch
import os
import math
import numpy as np
import subprocess
import random
import ref

def exec(opt, cacheFile):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data

    print("=> Generating list of data")
    with open(os.path.join(ref.data_root, 'train.txt'), 'r') as f:
        trainlst = f.read().splitlines()
    trainlst = [x[:-4] for x in trainlst]

    random.shuffle(trainlst)
    # set train/val 4800/200

    trainImg = [opt.data / "{}_rgb.png".format(x) for x in trainlst]
    trainLine = [opt.data / "{}_line.png".format(x) for x in trainlst]
    
    val_list = trainlst[:200]
    train_list = trainlst[200:]

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
