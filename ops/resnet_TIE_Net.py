"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
# from ops.tsm_util_stsm import tsm # STSM
# from ops.tsm_util import tsm  # TSM
from ops.tsm_util_astsm import AdaptionShift
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MEModule(nn.Module):
    """ Motion exciation module

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, reduction=8, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.bn2 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        # t feature
        reshape1_bottleneck = x.view((-1, self.n_segment) + x.size()[1:])  # n, t, c, h, w
        t_fea, __ = reshape1_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c, h, w
        t_fea = t_fea.contiguous().view((-1,) + t_fea.size()[2:]) # n*t-1, c, h, w
        # t+1 feature
        __, tPlusone_fea = reshape1_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c, h, w
        tPlusone_fea = tPlusone_fea.contiguous().view((-1,) + tPlusone_fea.size()[2:])  # n*t-1, c, h, w
        t_fea = self.avg_pool(t_fea) # n*t-1, c, 1, 1
        tPlusone_fea = self.avg_pool(tPlusone_fea) # n*t-1, c, 1, 1
        bottleneck1 = self.conv1(t_fea)
        bottleneck1 = self.bn1(bottleneck1) # n*t-1, c//r, 1, 1
        bottleneck2 = self.conv2(tPlusone_fea)
        bottleneck2 = self.bn2(bottleneck2) # n*t-1, c//r, 1, 1
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = bottleneck2 - bottleneck1  # n*t-1, c//r, 1, 1
        diff_fea = self.conv3(diff_fea) # n*t-1, c, 1, 1
        diff_fea = self.bn3(diff_fea)
        diff_fea = diff_fea.view((-1, self.n_segment-1) + diff_fea.size()[1:]) # n, t-1, c, 1, 1
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c, 1, 1
        diff_fea_pluszero = diff_fea_pluszero.contiguous().view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c, 1, 1
        y = self.sigmoid(diff_fea_pluszero)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder        
        self.num_segments = num_segments
        self.shift = AdaptionShift(inplanes, num_segments)
        self.me = MEModule(planes, reduction=8, n_segment=num_segments)


    def forward(self, x):
        identity = x
        # out = tsm(x, self.num_segments, 'zero')
        out = self.shift(x)  #在每一个残差快里插入tsm
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.me(out)  # ME模块在此。

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):

    def __init__(self, block, block2, layers, num_segments, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()          
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim=1)        
        self.num_segments = num_segments

       
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc1 = nn.Linear(512 * block.expansion, num_classes)                   
        self.fc = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)


        
    def _make_layer(self, block, planes, blocks, num_segments, stride=1):       
        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
            
        return nn.Sequential(*layers)            

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)    
        x = x.view(x.size(0), -1)
                       
        x = self.fc(x)
        return x


def resnet50(pretrained=False, shift='TSM',num_segments = 8):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments)
        # print(model)
    if pretrained:
        # model_path = '/home/lthpc/yanglinfeng/Asm_hsnet/pretrained/resnet50baseline.pth'
        model_path = model_zoo.load_url(model_urls['resnet50'])
        # model_path = 'pretrained/resnet50baseline.pth'
        # pretrained_dict1 = torch.load(model_path)
        new_state_dict2 = model.state_dict()
        for k, v in model_path.items():
            if (k in new_state_dict2) and "fc" not in k:  #这里改动了
                new_state_dict2.update({k: v})
            #     print ("%s层加载成功" % k)
            # else:
            #     print("%s层不加载" % k)
        model.load_state_dict(new_state_dict2)
    # print(model)
    return model

# def resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if (shift =='TSM'):
#         model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)
#     if pretrained:
#         model_path = '/home/lthpc/yanglinfeng/motionsqueeze_tif_tsm/pretrained/resnet50baseline.pth'
#         pretrained_dict = torch.load(model_path)
#         new_state_dict = model.state_dict()
#         for k, v in pretrained_dict.items():
#             if (k in new_state_dict) and k!="classifier.weight":
#                 new_state_dict.update({k: v})
#                 #                 print ("%s layer has pretrained weights" % k)
#         model.load_state_dict(new_state_dict)
#     return model



def resnet101(pretrained=False, shift='TSM',num_segments = 8):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 23, 3],num_segments=num_segments)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model
