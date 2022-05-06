import numpy as np
import torch
import torch.nn as nn


class Conv3DUnet(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv3d(in_channels=in_filters, out_channels=out_filters, kernel_size=self.kernel_size,
                                stride=1, padding=1, padding_mode='zeros', bias=False)
        self.batch_1 = nn.BatchNorm3d(num_features=out_filters)
        self.act_1 = nn.LeakyReLU()
        self.conv_2 = nn.Conv3d(in_channels=out_filters, out_channels=out_filters, kernel_size=self.kernel_size,
                                stride=1, padding=1, padding_mode='zeros', bias=False)
        self.batch_2 = nn.BatchNorm3d(num_features=out_filters)
        self.act_2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.act_2(x)
        return x


class Encoder3DUnet(nn.Module):
    def __init__(self, base_filters):
        super().__init__()
        self.base_filters = base_filters

        self.conv_d1 = Conv3DUnet(in_filters=1, out_filters=base_filters)
        self.pool_d1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv_d2 = Conv3DUnet(in_filters=base_filters, out_filters=base_filters * np.power(2, 1))
        self.pool_d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv_d3 = Conv3DUnet(in_filters=base_filters * np.power(2, 1), out_filters=base_filters * np.power(2, 2))
        self.pool_d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv_d4 = Conv3DUnet(in_filters=base_filters * np.power(2, 2), out_filters=base_filters * np.power(2, 3))

    def forward(self, x):
        conv_1 = self.conv_d1(x)
        pool1 = self.pool_d1(conv_1)
        conv_2 = self.conv_d2(pool1)
        pool2 = self.pool_d2(conv_2)
        conv_3 = self.conv_d3(pool2)
        pool3 = self.pool_d3(conv_3)
        conv_4 = self.conv_d4(pool3)
        return conv_1, conv_2, conv_3, conv_4


class Decoder3DUnet(nn.Module):
    def __init__(self, base_filters):
        super().__init__()
        self.up_u0 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.conv_u0 = Conv3DUnet(in_filters=base_filters * np.power(2, 3) + base_filters * np.power(2, 2),
                                  out_filters=base_filters * np.power(2, 2))
        self.up_u1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.conv_u1 = Conv3DUnet(in_filters=base_filters * np.power(2, 2) + base_filters * np.power(2, 1),
                                  out_filters=base_filters * np.power(2, 1))
        self.up_u2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.conv_u2 = Conv3DUnet(in_filters=base_filters * np.power(2, 1) + base_filters * np.power(2, 0),
                                  out_filters=base_filters)

    def forward(self, x):
        up3 = self.up_u0(x[3])
        concat4 = torch.cat([x[2], up3], axis=1)
        conv_up1 = self.conv_u0(concat4)
        up4 = self.up_u1(conv_up1)
        concat5 = torch.cat([x[1], up4], axis=1)
        conv_up2 = self.conv_u1(concat5)
        up6 = self.up_u2(conv_up2)
        concat7 = torch.cat([x[0], up6], axis=1)
        return self.conv_u2(concat7)


class Classification3DUnet(nn.Module):
    def __init__(self, base_filters):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=base_filters, out_channels=1, kernel_size=1,
                              stride=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        conv_c = self.conv(x)
        return conv_c  # self.act(conv_c)


class Classification3DUnetMulti(Classification3DUnet):
    def __init__(self, base_filters, n_classes):
        super(Classification3DUnetMulti, self).__init__(base_filters)
        self.conv = nn.Conv3d(in_channels=base_filters, out_channels=n_classes, kernel_size=1,
                              stride=1, padding=0)
