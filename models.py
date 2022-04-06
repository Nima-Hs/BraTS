import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_c=4, out_c=2, useBN=True, drop_rate=0, useSM=True):
        super(UNet, self).__init__()
        self.drop_rate = drop_rate
        self.out_c = out_c
        self.useSM = useSM
        self.conv1 = self.conv3x3(
            in_c, 64, useBN=useBN, drop_rate=self.drop_rate)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = self.conv3x3(
            64, 128, useBN=useBN, drop_rate=self.drop_rate)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = self.conv3x3(
            128, 256, useBN=useBN, drop_rate=self.drop_rate)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = self.conv3x3(
            256, 512, useBN=useBN, drop_rate=self.drop_rate)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv5 = self.conv3x3(
            512, 1024, useBN=useBN, drop_rate=self.drop_rate)
        self.upconv1 = self.upsample(1024, 512, drop_rate=self.drop_rate)
        self.conv6 = self.conv3x3(
            1024, 512, useBN=useBN, drop_rate=self.drop_rate)
        self.upconv2 = self.upsample(512, 256, drop_rate=self.drop_rate)
        self.conv7 = self.conv3x3(
            512, 256, useBN=useBN, drop_rate=self.drop_rate)
        self.upconv3 = self.upsample(256, 128, drop_rate=self.drop_rate)
        self.conv8 = self.conv3x3(
            256, 128, useBN=useBN, drop_rate=self.drop_rate)
        self.upconv4 = self.upsample(128, 64, drop_rate=self.drop_rate)
        self.conv9 = self.conv3x3(
            128, 64, useBN=useBN, drop_rate=self.drop_rate)

        if self.useSM:
            self.convlast = nn.Sequential(
                nn.Conv2d(64, self.out_c, kernel_size=3, stride=1,
                        padding=1, padding_mode='reflect'),
                nn.Dropout2d(p=self.drop_rate),
                nn.ReLU(inplace=True)
            )
        else:
            self.convlast = nn.Sequential(
                nn.Conv2d(64, self.out_c, kernel_size=3, stride=1,
                        padding=1, padding_mode='reflect'),
                nn.Dropout2d(p=self.drop_rate),
                nn.Sigmoid()
            )

    def conv3x3(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                bias=True, useBN=False, drop_rate=0):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride,
                          padding=padding, bias=bias, padding_mode='reflect'),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size, stride,
                          padding=padding, bias=bias, padding_mode='reflect'),
                nn.BatchNorm2d(out_c),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride,
                          padding=padding, bias=bias, padding_mode='reflect'),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size, stride,
                          padding=padding, bias=bias, padding_mode='reflect'),
                nn.Dropout2d(p=drop_rate),
                nn.ReLU(inplace=True)
            )

    def upsample(self, in_c, out_c, bias=True, drop_rate=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               stride=2, padding=0, bias=bias),
            nn.Dropout2d(p=drop_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, example):
        x1 = self.conv1(example)
        x2 = self.conv2(self.maxpool1(x1))
        x3 = self.conv3(self.maxpool2(x2))
        x4 = self.conv4(self.maxpool3(x3))
        x5 = self.upconv1(self.conv5(self.maxpool4(x4)))

        x6 = self.upconv2(self.conv6(torch.cat((x5, x4), 1)))
        x7 = self.upconv3(self.conv7(torch.cat((x6, x3), 1)))
        x8 = self.upconv4(self.conv8(torch.cat((x7, x2), 1)))
        x9 = self.convlast(self.conv9(torch.cat((x8, x1), 1)))
#         Complete = F.softmax(x9[:,0:2], dim=1)
#         Core = F.softmax(x9[:, 2:4], dim=1)
#         Enhancing = F.softmax(x9[:, 4:], dim=1)
#         return torch.cat((Complete, Core, Enhancing), 1)
        if self.useSM:
            return F.softmax(x9, dim=1)
        else:
            return x9
