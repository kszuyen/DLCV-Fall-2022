import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class vgg16_fcn32(nn.Module):
    def __init__(self):
        super(vgg16_fcn32, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 3, 18),
            nn.ReLU(),
            nn.Conv2d(3, 3, 18),
            nn.ReLU(),
            nn.Conv2d(3, 3, 16),
            nn.ReLU(),
            nn.Conv2d(3, 3, 16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg16 = nn.Sequential(*list(self.vgg16.children())[:-2])
        # feature size = 512*7*7
        self.upsamplex32 = nn.ConvTranspose2d(
            512, 7, kernel_size=64, stride=32)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv_block(x)
        # outsize = 2*224*224
        x = self.vgg16(x)
        # outsize = 512*7*7
        x = self.upsamplex32(x)
        # outsize = 7*256*256
        x = self.upsamplex2(x)
        # outsize = 7*512*512

        return x


class vgg16_fcn8(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()
        # self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)

        feats = list(vgg16(weights=VGG16_Weights.DEFAULT).features.children())

        self.feat1 = nn.Sequential(*feats[0:5])
        self.feat2 = nn.Sequential(*feats[5:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])
        self.feat6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
        )

        self.upsamplex64 = nn.ConvTranspose2d(
            1024, num_classes, kernel_size=64, stride=64)
        self.upsamplex32 = nn.ConvTranspose2d(
            512, num_classes, kernel_size=32, stride=32)
        self.upsamplex16 = nn.ConvTranspose2d(
            512, num_classes, kernel_size=16, stride=16)
        self.upsamplex8 = nn.ConvTranspose2d(
            256, num_classes, kernel_size=8, stride=8)
        self.upsamplex4 = nn.ConvTranspose2d(
            128, num_classes, kernel_size=4, stride=4)
        self.upsamplex2 = nn.ConvTranspose2d(
            64, num_classes, kernel_size=2, stride=2)
        # self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # x = self.conv_block(x)
        # 3*512*512
        feat1 = self.feat1(x)
        # 64*256*256
        feat2 = self.feat2(feat1)
        # 128*128*128
        feat3 = self.feat3(feat2)
        # 256*64*64
        feat4 = self.feat4(feat3)
        # 512*32*32
        feat5 = self.feat5(feat4)
        # 512*16*16
        feat6 = self.feat6(feat5)
        # 1024*8*8
        score = self.upsamplex64(feat6) + self.upsamplex32(feat5) + \
            self.upsamplex16(feat4) + self.upsamplex8(feat3) + \
            self.upsamplex4(feat2) + self.upsamplex2(feat1)
        # score = self.upsamplex2(score)

        return score


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1,
                      1, bias=False),  # using batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1,
                      1, bias=False),  # using batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=7, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of Unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of Unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # if x.shape!=skip_connections.shape:
            #     x = TF.resize(x, size=skip_connections.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        return x

