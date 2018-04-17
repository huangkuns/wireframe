import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['Inception2', 'inception_v2']

def inception_v2(pretrained=False, **kwargs):
    r"""Inception v2 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception2(**kwargs)
        #model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        raise ValueError
        return model

    return Inception2(**kwargs)


class Inception2(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, with_bn = True,  transform_input=False):
        super(Inception2, self).__init__()
        #self.aux_logits = aux_logits
        self.transform_input = transform_input
        # for some task, batch_size is 1 or so, bn might be of no benefit.
        self.with_bn = with_bn
        if with_bn:
            print("Inception_v2 use batch norm")
        else:
            print("Inception_v2 not use batch norm")

        self.Conv2d_1a_7x7 = BasicConv2d(3, 64, with_bn=with_bn, kernel_size=7, stride=2, padding=3)
        self.Conv2d_2b_1x1 = BasicConv2d(64, 64, with_bn=with_bn, kernel_size=1)
        self.Conv2d_2c_3x3 = BasicConv2d(64, 192, with_bn=with_bn,  kernel_size=3, stride=1, padding=1)
        self.Mixed_3b = InceptionD(192, pool_features=32)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #        import scipy.stats as stats
        #        stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #        X = stats.truncnorm(-2, 2, scale=stddev)
        #        values = torch.Tensor(X.rvs(m.weight.data.numel()))
        #        values = values.view(m.weight.data.size())
        #        m.weight.data.copy_(values)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x):
        x = self.Conv2d_1a_7x7(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.Conv2d_2b_1x1(x)
        x = self.Conv2d_2c_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.Mixed_3b(x)
        
        return x


class InceptionD(nn.Module):

    def __init__(self, in_channels, pool_features, with_bn = False):
        super(InceptionD, self).__init__()
        self.branch0_1x1 = BasicConv2d(in_channels, 64, with_bn = with_bn, kernel_size=1)

        self.branch1_3x3_1 = BasicConv2d(in_channels, 64, with_bn=with_bn, kernel_size=1)
        self.branch1_3x3_2 = BasicConv2d(64, 64, with_bn = with_bn, kernel_size=3, padding=1)

        self.branch2_3x3_1 = BasicConv2d(in_channels, 64, with_bn=with_bn, kernel_size=1)
        self.branch2_3x3_2 = BasicConv2d(64, 96, with_bn=with_bn, kernel_size=3, padding=1)
        self.branch2_3x3_3 = BasicConv2d(96, 96, with_bn=with_bn, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, with_bn=with_bn, kernel_size=1)

    def forward(self, x):
        branch0 = self.branch0_1x1(x)

        branch1 = self.branch1_3x3_1(x)
        branch1 = self.branch1_3x3_2(branch1)

        branch2= self.branch2_3x3_1(x)
        branch2 = self.branch2_3x3_2(branch2)
        branch2 = self.branch2_3x3_3(branch2)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch0, branch1, branch2, branch_pool]
        return torch.cat(outputs, 1)



class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, with_bn=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.00001)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

def loadHype(hypefilename):
    import json
    with open(hypefilename, 'r') as f:
        data = json.load(f)
        return data 


if __name__ ==  "__main__":
    H = loadHype('../8.1.json')
    model = inception_v2(pretrained=False, H=H)
    print(model)
