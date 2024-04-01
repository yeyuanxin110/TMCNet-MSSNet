# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

# import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab_utils import ResNet
from models.deeplab_utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplab_utils.encoder import Encoder
from models.deeplab_utils.net_utils import FeatureRectifyModule as FRM
from models.deeplab_utils.net_utils import FeatureFusionModule as FFM

class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x*wei

class iAFF(nn.Module):


    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att_2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att_4 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2_2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2_4 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x, residual):

        xl=self.local_att(x)
        xl2=self.local_att2(residual)
        xg=self.global_att(x)
        xg2=self.global_att2(residual)

        l=xl+0.5*xl2
        g=xg+0.5*xg2
        xlg=l+g
        xi=self.sigmoid(xlg)
        x0=x*xi+residual*(1-xi)



        return x0


class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(
            x,
            size=low_level_feature.size()[2:4],
            mode='bilinear',
            align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self,
                 num_classes=2,
                 in_channels=3,
                 arch='resnet101',
                 output_stride=16,
                 bn_momentum=0.9,
                 freeze_bn=False,
                 pretrained=False,
                 norm_fuse=nn.BatchNorm2d,
                 **kwargs):
        super(DeepLab, self).__init__(**kwargs)
        self.model_name = 'deeplabv3plus_' + arch

        # Setup arch
        if arch == 'resnet18':
            NotImplementedError('resnet18 backbone is not implemented yet.')
        elif arch == 'resnet34':
            NotImplementedError('resnet34 backbone is not implemented yet.')
        elif arch == 'resnet50':
            self.backbone = ResNet.resnet50(bn_momentum, pretrained)
            self.backbone_sar = ResNet.resnet50(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif arch == 'resnet101':
            self.backbone = ResNet.resnet101(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(num_classes, bn_momentum)
        self.iaff = iAFF(channels=2048, r=4)
        self.iaff_low = iAFF(channels=256, r=4)
        self.MS_CAM_1 = MS_CAM(channels=2048, r=4)
        self.MS_CAM_low_1 = MS_CAM(channels=256, r=4)
        self.MS_CAM_2 = MS_CAM(channels=2048, r=4)
        self.MS_CAM_low_2 = MS_CAM(channels=256, r=4)
        self.MS_CAM = MS_CAM(channels=256, r=4)
        self.MS_CAM_low = MS_CAM(channels=256, r=4)
        self.FRM1=FRM(dim=2048,reduction=1)
        self.FFM1=FFM(dim=2048,reduction=1,num_heads=1,norm_layer=norm_fuse)
        self.FRM2 = FRM(dim=256, reduction=1)
        self.FFM2 = FFM(dim=256, reduction=1, num_heads=1, norm_layer=norm_fuse)


    def forward(self, input,input_sar):
        x, low_level_features = self.backbone(input)
        x_sar, low_level_features_sar = self.backbone(input_sar)
        x_sar=0.1*x_sar
        low_level_features_sar = 0.1 * low_level_features_sar
        x,x_sar=self.FRM1(x,x_sar)
        x=self.FFM1(x,0.1*x_sar)
        low_level_features, low_level_features_sar = self.FRM2(low_level_features, low_level_features_sar)
        low_level_features = self.FFM2(low_level_features, 0.1*low_level_features_sar)
        x = self.encoder(x)
        x=self.MS_CAM(x)
        low_level_features=self.MS_CAM_low(low_level_features)
        predict = self.decoder(x, low_level_features)
        output = F.interpolate(
            predict,
            size=input.size()[2:4],
            mode='bilinear',
            align_corners=True)
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


class Decoder_without_last_conv(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder_without_last_conv, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        # self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        # self.dropout3 = nn.Dropout(0.1)
        # self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(
            x,
            size=low_level_feature.size()[2:4],
            mode='bilinear',
            align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        # x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        # x_4_cat = self.dropout3(x_4_cat)
        # x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def build_model(num_classes=5, in_channels=3,pretrained=False,arch='resnet101'):
    model = DeepLab(num_classes=num_classes, in_channels=in_channels,pretrained=pretrained,arch=arch)
    return model

if __name__ == "__main__":
    model = DeepLab(
        output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
