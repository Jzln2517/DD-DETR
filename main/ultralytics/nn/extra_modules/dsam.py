import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



__all__ = ['ConvBNReLU', 'DSAM', 'Pred_Layer', 'FF','BF','ASPP']   # 这个部分必须有声明到全局路径中去

BatchNorm2d = nn.BatchNorm2d
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.bn.train(True)
        self.bn.track_running_stats = False
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Pred_Layer(nn.Module):
    def __init__(self, in_c=2048):  # 修改默认输入通道为2048
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),  # 2048 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),  # 256 -> 1
        )

    def forward(self, x):
        x = self.enlayer(x)   # 这一步之后是得到的E0
        x1 = self.outlayer(x) # 这一步之后得到的应该是
        return x, x1
# FF
class FF(nn.Module):
    def __init__(self, in_c):
        super(FF, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        enhanced_feat, new_pred = self.rgbd_pred_layer(ff_feat)
        return enhanced_feat, new_pred


# BF
class BF(nn.Module):
    def __init__(self, in_c):
        super(BF, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, 32, 1)
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(bf_feat)
        return enhanced_feat, new_pred


# ASPP for DSAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()
        print("------test in_c")
        print(in_c)
        in_c = in_c[0]    # 本来没有这一行代码，但是因为输入数据是p0和p4，模型会自动统计这两个特征图的通道数然后合并成一个列表传过来，因此这里取第一个传过来的特征图的通道数。
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c , 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        print("x4---")
        print(x4.shape)
        print(x4.size)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class DSAM(nn.Module):
    def __init__(self, in_c):  # in_c应该是feat的输入通道数512
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)  # 前景区ASPP，输入512，输出512
        self.bf_conv = ASPP(in_c)  # 背景区ASPP，输入512，输出512
        self.rgbd_pred_layer = Pred_Layer(2048)  # 修改为512*2因为我们将拼接两个512通道的特征
        self.channel_adjuster = nn.Conv2d(
            in_channels=2048,  # 输入通道数改为2048
            out_channels=512,  # 输出通道数保持512以匹配feat的通道数
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x_list):
        feat = x_list[0]  # 输入特征，通道数512
        pred = x_list[1]  # 预测特征，通道数2048
        [_, _, H, W] = feat.size()

        # 调整pred的尺寸和通道数
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        pred = self.channel_adjuster(pred)  # 2048->512

        # 前背景特征提取
        ff_feat = self.ff_conv(feat * pred)  # 前景区特征
        bf_feat = self.bf_conv(feat * (1 - pred))  # 背景区特征,拼接后通道数变为1024
        print("test ---- ff_Feat")
        print(ff_feat.size)
        print(ff_feat.shape)
        print(bf_feat.size)
        print(bf_feat.shape)
        # 特征增强
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))  # 两个1024的特征图拼接后变为2048通道数

        return enhanced_feat  # 返回增强后的特征，通道数保持512