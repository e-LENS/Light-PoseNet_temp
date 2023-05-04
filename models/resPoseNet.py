import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

### Pretrained ###
model_urls = {
    'resnet18im': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34im': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50im': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101im': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152im': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


###############################################################################
# Functions
###############################################################################

def weight_init_resnet(key, module, weights=None):

    if weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "WPQR":
            init.normal_(module.weight.data, 0.0, 0.01)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key + "_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key + "_0").encode()])
    return module


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(input_nc, model, pretrained=True , init_from=None, isKD=False, isTest=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    if model == 'resnet18':
        netG = resnet18im(pretrained=pretrained, progress = True, isKD=isKD, isTest=isTest)

    elif model == 'resnet34':
        netG = resnet34im(pretrained=pretrained, progress = True, isKD=isKD, isTest=isTest)

    elif model == 'resnet50':
        netG = resnet50im(pretrained=pretrained, progress = True, isKD=isKD, isTest=isTest)

    elif model == 'resnet101':
        netG = resnet101im(pretrained=pretrained, progress = True, isKD=isKD, isTest=isTest)

    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


# defines the regression module for resnet
class Regression(nn.Module):
    def __init__(self, weights=None, input_cls=512, mid_cls=1024):
        super(Regression, self).__init__()

        dropout_rate = 0.5
        #dropout_rate = 0.7
        #input_cls = [512, 2048]

        self.dropout = nn.Dropout(p=dropout_rate)

        #self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
        self.cls_fc_pose = nn.Sequential(*[weight_init_resnet("pose", nn.Linear(input_cls, mid_cls)),
                                           nn.ReLU(inplace=True)])
        self.cls_fc_xy = weight_init_resnet("XYZ", nn.Linear(mid_cls, 3), weights=weights)
        self.cls_fc_wpqr = weight_init_resnet("WPQR", nn.Linear(mid_cls, 4), weights=weights)


    def forward(self, input):
        #output = self.projection(input)
        #output = self.cls_fc_pose(output.view(output.size(0), -1))
        output = self.cls_fc_pose(input.view(input.size(0), -1))
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]


###############################################
#              ResNet_PoseNet                 #
###############################################
# Implementation Link 1 : https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
# Implementation Link 2 : Attention feature distillation

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []
        x = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out, features + [out]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []
        x = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out, features + [out]


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 isKD=False, isTest=False
                 ):
        super(ResNet, self).__init__()

        self.isKD = isKD

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        f0 = x
        # x = self.relu(x)
        x = self.maxpool(x)

        x, f1 = self.layer1(x)
        f1_act = [self.relu(f) for f in f1]
        x, f2 = self.layer2(x)
        f2_act = [self.relu(f) for f in f2]
        x, f3 = self.layer3(x)
        f3_act = [self.relu(f) for f in f3]
        x, f4 = self.layer4(x)
        f4_act = [self.relu(f) for f in f4]
        x = self.avgpool(self.relu(x))
        x = torch.flatten(x, 1)   # torch.flatten 부분 없앨지 상의
        f5 = x
        x = self.fc(x)
        if self.isKD:
            return [self.relu(f0)] + f1_act + f2_act + f3_act + f4_act + [f5], x
        else:
            return x


def _resnet(arch, block, layers, pretrained, progress, isKD=False, isTest=False):
    model = ResNet(block, layers, isKD=isKD, isTest=isTest)
    if pretrained:
        print("ResNet initialized with ImageNet Pretrained model.")
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model


def resnet18im(pretrained=False, progress=True, isKD=False, isTest=False):
    # Pretrained = True : ImageNet

    resnet18 = _resnet('resnet18im', BasicBlock, [2, 2, 2, 2], pretrained=pretrained, progress=progress, isKD=isKD, isTest=isTest)
    resnet18.fc = Regression()

    return resnet18


def resnet34im(pretrained=False, progress=True, isKD=False, isTest=False):
    # Pretrained = True : ImageNet
    resnet34 = _resnet('resnet34im', BasicBlock, [3, 4, 6, 3], pretrained, progress, isKD)
    resnet34.fc = Regression()

    return resnet34


def resnet50im(pretrained=False, progress=True, isKD=False, isTest=False):
    # Pretrained = True : ImageNet
    resnet50 = _resnet('resnet50im', Bottleneck, [3, 4, 6, 3], pretrained, progress, isKD)
    resnet50.fc = Regression(input_cls=2048)

    return resnet50


def resnet101im(pretrained=False, progress=True, isKD=False, isTest=False):
    # Pretrained = True : ImageNet
    resnet101 = _resnet('resnet101im', Bottleneck, [3, 4, 23, 3], pretrained, progress, isKD)
    resnet101.fc = Regression(input_cls=2048)

    return resnet101



