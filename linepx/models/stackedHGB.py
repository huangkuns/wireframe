from .layers.Residual import Residual
import torch
import torch.nn as nn

class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2

class HourglassNet3D(nn.Module):
    def __init__(self, nStack, nModules, nFeats, nOutChannels):
        super(HourglassNet3D, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.nOutChannels = nOutChannels
        self.conv1_ = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1),
                                                    nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.nOutChannels, bias = True, kernel_size = 1, stride = 1))
            _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
            _tmpOut_.append(nn.Conv2d(self.nOutChannels, self.nFeats, bias = True, kernel_size = 1, stride = 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

        self.deconv1 = nn.ConvTranspose2d(self.nOutChannels, self.nOutChannels//2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nOutChannels//2)
        self.deconv2 = nn.ConvTranspose2d(self.nOutChannels//2, self.nOutChannels//4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nOutChannels//4)
        self.conv2 = nn.Conv2d(self.nOutChannels//4, 1, kernel_size=5, stride=1, padding=2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, line):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)

            ll_ = self.ll_[i](ll)
            tmpOut_ = self.tmpOut_[i](tmpOut)
            x = x + ll_ + tmpOut_

        shareFeat = out[-1]
        lineOut = self.relu(self.bn2(self.deconv1(shareFeat)))
        lineOut = self.relu(self.bn3(self.deconv2(lineOut)))
        lineOut = self.conv2(lineOut)

        line_loss = nn.MSELoss()(lineOut, line)
        loss = line_loss

        return loss, line_loss, lineOut



def createModel(opt):
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nOutChannels)

    return model
