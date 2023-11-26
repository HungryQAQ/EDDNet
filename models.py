import math
import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.downsample = downsample
        self.conv11=nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=True),
        )
        self.conv22=nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.downsample is not None:
            x=self.conv11(x)
            out = out + x
            
        out_r = self.relu(out)  # out_r

        out = self.conv2(out_r)
        if self.downsample is not None:
            out_r=self.conv22(out_r)
            out = out + out_r
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out




class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1)).permute(0, 2,
                                                                                                        1).reshape(b, c,
                                                                                                                   1,
                                                                                                                   1)
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out


class EDDNet(nn.Module):
    def __init__(self, block, num_features=64):
        super(EDDNet, self).__init__()
        self.inplanes = num_features

        self.conv = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer1 = self._make_layer(block, 64, 128, 3, stride=2)
        self.layer2 = self._make_layer(block, 128, 256, 3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.layer3 = self._make_layer(block, 256, 512, 4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        self.layer4 = self._make_layer(block, 256, 256, 3)
        self.layer5 = self._make_layer(block, 128, 128, 3)
        self.layer6 = self._make_layer(block, 128, 128, 3)
        self.conv2 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.PReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.PReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.PReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=True),
                                   nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True),
                                   )


        self.dcd1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dcd11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dcdconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)

        self.dcd2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dcd22 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dcdconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)

        self.dcd3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ↑#
        self.ca3 = SequentialPolarizedSelfAttention(512)
        self.ca2 = SequentialPolarizedSelfAttention(256)
        self.ca1 = SequentialPolarizedSelfAttention(192)

        self.ca = SequentialPolarizedSelfAttention(128)
        self.lastconv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(1.0 / n))
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        self.inplanes = inplanes
        if stride != 1:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=2, bias=True),
            )
       
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv(x)

        res2 = self.layer1(res)
        res3 = self.layer2(res2)
        out = self.layer3(res3)
        out = self.deconv1(out)
        out = self.layer4(out)
        out1 = torch.cat((out, res3), dim=1)
        out1 = self.ca3(out1)

        dcd1 = self.dcd1(out1)
        dcd1 = torch.cat((dcd1, res3), dim=1)
        dcd1 = self.dcd11(dcd1)
        dcd1 = self.dcdconv1(dcd1)

        out = self.deconv2(out1)
        out = self.layer5(out)
        out2 = torch.cat((out, res2), dim=1)
        out2 = self.ca2(out2)

        dcd2 = self.dcd2(out2)
        dcd2 = torch.cat((dcd2, res2, dcd1), dim=1)
        dcd2 = self.dcd22(dcd2)
        dcd2 = self.dcdconv2(dcd2)

        out = self.deconv3(out2)
        out = self.layer6(out)
        out3 = torch.cat((out, res), dim=1)
        out3 = self.ca1(out3)

        dcd3 = self.dcd3(out3)
        dcd3 = torch.cat((dcd3, res, dcd2), dim=1)

        out = self.conv2(dcd3)
        out = self.ca(out)
        out = self.lastconv(out)

        return x - out
