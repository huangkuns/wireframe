import os
import cv2
import numpy as np
import time
from torch.autograd import Variable
import torch.optim as optim
import util.visualize as vis
from util.progbar import progbar

class stackHourglassTrainer():
    def __init__(self, model, criterion, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        self.opt = opt

        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening, weight_decay=opt.weightDecay)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.momentum, 0.999), eps=1e-8, weight_decay=opt.weightDecay)

        if self.optimState is not None:
            self.optimizer.load_state_dict(self.optimState)

        self.logger = {'train' : open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val' : open(os.path.join(opt.resume, 'test.log'), 'a+')}

    def train(self, trainLoader, epoch):
        self.model.train()

        print('=> Training epoch # ' + str(epoch))

        avgLoss = 0
        visImg = []

        self.progbar = progbar(len(trainLoader), width=self.opt.barwidth)

        for i, (inputData, line, imgids) in enumerate(trainLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()

            inputData_var, line_var = Variable(inputData), Variable(line)
            self.optimizer.zero_grad()
            if self.opt.GPU:
                inputData_var = inputData_var.cuda()
                line_var = line_var.cuda()
            dataTime = time.time() - start

            loss, line_loss, line_result = self.model.forward(inputData_var, line_var)

            loss.backward()
            self.optimizer.step()
            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)

            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(trainLoader), runTime, dataTime, loss.data[0])
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0])])

            if i <= self.opt.visTrain:
                visImg.append(inputData)
                visImg.append(line_result.cpu().data)
                visImg.append(line)

            #if i == self.opt.visTrain:
            #    self.visualize(visImg, epoch, 'train', trainLoader.dataset.postprocess, trainLoader.dataset.postprocessLine)

        log = '\n * Finished training epoch # %d     Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['train'].write(log)
        print(log)

        return avgLoss

    def test(self, valLoader, epoch):
        self.model.eval()

        avgLoss = 0
        visImg = []

        self.progbar = progbar(len(valLoader), width=self.opt.barwidth)

        for i, (inputData, line, imgids) in enumerate(valLoader):
            if self.opt.debug and i > 10:
                break

            start = time.time()

            inputData_var, line_var = Variable(inputData), Variable(line)
            if self.opt.GPU:
                inputData_var = inputData_var.cuda()
                line_var = line_var.cuda()
            dataTime = time.time() - start

            loss, line_loss, line_result = self.model.forward(inputData_var, line_var)

            runTime = time.time() - start

            avgLoss = (avgLoss * i + loss.data[0]) / (i + 1)

            log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f\n' % (epoch, i, len(valLoader), runTime, dataTime, loss.data[0])
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('Loss', loss.data[0])])

            if i <= self.opt.visTest:
                visImg.append(inputData.cpu())
                visImg.append(line_result.cpu().data)
                visImg.append(line)

            if i == self.opt.visTest:
                self.visualize(visImg, epoch, 'test', valLoader.dataset.postprocess, valLoader.dataset.postprocessLine)

            outDir = os.path.join(self.opt.resume, str(epoch))
            if not os.path.exists(outDir):
                os.makedirs(outDir)

            for j in range(len(imgids)):
                np.save(os.path.join(outDir, imgids[j] + '_line.npy'), valLoader.dataset.postprocessLine()(line_result.cpu().data[j].numpy()))

        log = '\n * Finished testing epoch # %d      Loss: %1.4f\n' % (epoch, avgLoss)
        self.logger['val'].write(log)
        print(log)

        return avgLoss

    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()

    def visualize(self, visImg, epoch, split, postprocess, postprocessLine):
        outputImgs = []
        for i in range(len(visImg) // 3):
            for j in range(self.opt.batchSize):
                outputImgs.append(postprocess()(visImg[3 * i][j].numpy()))
                outputImgs.append(postprocessLine()(visImg[3 * i + 1][j].numpy()))
                outputImgs.append(postprocessLine()(visImg[3 * i + 2][j].numpy()))
        vis.writeImgHTML(outputImgs, epoch, split, 3, self.opt)

    def visJunc(self, img, junc, opt):
        juncConf, juncRes, juncBinConf, juncBinRes = junc
        juncConf, juncRes, juncBinConf, juncBinRes = juncConf.numpy(), juncRes.numpy(), juncBinConf.numpy(), juncBinRes.numpy()
        imgDim = opt.imgDim
        thres = opt.visThres
        out = img.astype(np.uint8).copy()
        blockSize = 8
        for i in range(imgDim // blockSize):
            for j in range(imgDim // blockSize):
                if juncConf[i][j] > thres:
                    p = np.array((i, j)) * np.array((blockSize, blockSize))
                    p = p + (juncRes[:, i, j] + 0.5) * np.array((blockSize, blockSize))
                    for t in range(12):
                        if juncBinConf[t, i, j] > thres:
                            theta = (t + juncBinRes[t, i, j] + 0.5) * 30
                            theta = theta * np.pi / 180
                            end = p + 20 * np.array((np.cos(theta), np.sin(theta)))
                            out = cv2.line(out, self.regInt(p), self.regInt(end), (0, 255, 0), 2)
        return out

    def regInt(self, x):
        return (int(round(x[0])), int(round(x[1])))

def createTrainer(model, criterion, opt, optimState):
    return stackHourglassTrainer(model, criterion, opt, optimState)
