# unet 13.39M

#GPT:
#在 down4 层之后的上采样过程中，通道数较少（512, 256, 128, 64），这减少了卷积操作和参数数量


import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, img_ch, output_ch, bilinear=True):
        super(UNet, self).__init__()
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.bilinear = bilinear

        #从 n_channels 到 64，再依次到 128、256、512，然后保持在 512，通过上采样再返回到 64。
        self.inc = DoubleConv(img_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear) #与下采样的对应层进行跳跃连接，256+256，故下次层输入为512 lhh
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) #最后一层，瓶颈层
        x = self.up2(x, x3)  #与x3进行跳跃连接
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#------------------- Parts of the U-Net model ------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),#通道数改变，
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),#通道数不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    #扩展路径中的每一步都用2×2上卷积对特征映射进行上采样。

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #scale_factor=2 表示将特征图的宽度和高度都放大两倍。
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            # ConvTranspose2d函数：该函数是用来进行转置卷积的，它主要做了这几件事：
            # 首先，对输入的feature map进行padding操作，得到新的feature map；
            # 然后，随机初始化一定尺寸的卷积核；
            # 最后，用随机初始化的一定尺寸的卷积核在新的feature map上进行卷积操作。
            # 当s=1时，对于原feature map不进行插值操作，只进行padding操作；
            # 当s>1是还要进行插值操作，也就是这里采用的上采样操作

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # 低分辨率的特征图（x1）通过上采样与高分辨率的特征图（x2）融合
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)