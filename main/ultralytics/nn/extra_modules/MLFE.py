import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MLFE']   # 这个部分必须有声明到全局路径中去

class MLFE(nn.Module):
    def __init__(self, target_level, p2_channels=64, reduction_ratio=4):
        """
        多级特征增强模块 (适配不同层级的通道数)

        参数:
            p2_channels: P2层的通道数(64)
            reduction_ratio: 注意力机制中的通道缩减比例
        """
        super(MLFE, self).__init__()
        p2_channels = p2_channels[0]
        print("test------args------")
        print(target_level)
        print(p2_channels)
        print(reduction_ratio)
        self.target_level = target_level
        self.p2_channels = p2_channels
        self.reduction_ratio = reduction_ratio

        # 通道注意力模块 (针对P2的64通道)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(p2_channels, p2_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(p2_channels // reduction_ratio, p2_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力模块 (共享权重)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 用于调整P2特征图尺寸和通道数的卷积 (针对不同层级有不同的处理)
        self.adjust_convs = nn.ModuleDict({
            'p3': nn.Sequential(
                nn.Conv2d(p2_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            'p4': nn.Sequential(
                nn.Conv2d(p2_channels, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            'p5': nn.Sequential(
                nn.Conv2d(p2_channels, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        })

        # 特征增强卷积 (针对不同层级有不同的处理)
        self.enhance_convs = nn.ModuleDict({
            'p3': nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1)
            ),
            'p4': nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            ),
            'p5': nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1)
            )
        })

        # 残差连接前的1x1卷积 (确保通道数匹配)
        self.residual_convs = nn.ModuleDict({
            'p3': nn.Conv2d(128, 128, kernel_size=1),
            'p4': nn.Conv2d(256, 256, kernel_size=1),
            'p5': nn.Conv2d(512, 512, kernel_size=1)
        })
        self.channel_128_to_512 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.channel_256_to_512 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_list):
        """
        前向传播

        参数:
            p2_feat: P2特征图 [B, 64, H, W]
            high_level_feat: 高级特征图 (P3/P4/P5)
                            [B, 128/256/512, H/s, W/s]
            target_level: 目标层级 ('p3', 'p4' 或 'p5')

        返回:
            增强后的特征图 (与high_level_feat相同尺寸和通道数)
        """
        p2_feat = feat_list[0]
        high_level_feat = feat_list[1]
        # 1. 调整P2特征图尺寸和通道数以匹配高级特征图
        adjusted_p2 = F.interpolate(
            p2_feat,
            size=high_level_feat.size()[2:],
            mode='bilinear',
            align_corners=True
        )
        print(self.target_level)
        adjusted_p2 = self.adjust_convs[self.target_level](adjusted_p2)

        # 2. 通道注意力 (在P2上计算，然后扩展到目标通道数)
        channel_weights = self.channel_attention(p2_feat)

        # 计算需要的重复倍数
        target_channels = high_level_feat.size(1)
        repeat_factor = target_channels // self.p2_channels



        # 将注意力权重扩展到目标通道数
        if self.target_level == 'p3':
            channel_weights = channel_weights.repeat(1, 2, 1, 1)  # 64->128
        elif self.target_level == 'p4':
            channel_weights = channel_weights.repeat(1, 4, 1, 1)  # 64->256
        else:  # p5
            channel_weights = channel_weights.repeat(1, 8, 1, 1)  # 64->512

        # 调整注意力权重尺寸
        channel_weights = F.interpolate(
            channel_weights,
            size=high_level_feat.size()[2:],
            mode='bilinear',
            align_corners=True
        )
        print("------------channel-----------")
        print(high_level_feat.size)
        print(high_level_feat.shape)
        print(channel_weights.size)
        print(channel_weights.shape)
        # 调整channel_weights的通道数
        if self.target_level == 'p3':
            channel_weights = self.channel_128_to_512(channel_weights)
        # 应用通道注意力
        channel_enhanced = high_level_feat * channel_weights

        # 3. 空间注意力
        avg_pool = torch.mean(channel_enhanced, dim=1, keepdim=True)
        max_pool = torch.max(channel_enhanced, dim=1, keepdim=True)[0]
        spatial_weights = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_weights)
        spatial_enhanced = channel_enhanced * spatial_weights

        # 4. 特征融合 (调整后的P2特征 + 注意力增强的高级特征)
        fused_feat = adjusted_p2 + spatial_enhanced

        # 5. 特征增强
        enhanced_feat = self.enhance_convs[self.target_level](fused_feat)

        # 6. 残差连接
        residual = self.residual_convs[self.target_level](high_level_feat)
        output = F.relu(enhanced_feat + residual)

        return output
