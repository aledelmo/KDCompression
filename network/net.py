import torch.nn as nn
from .layers import Encoder3DUnet, Decoder3DUnet, Classification3DUnet, Classification3DUnetMulti


class UNet3D(nn.Module):
    def __init__(self, base_filters=32):
        super().__init__()
        self.base_filters = base_filters

        self.encoder = Encoder3DUnet(base_filters)
        self.decoder = Decoder3DUnet(base_filters)
        self.classification = Classification3DUnet(base_filters)

    def forward(self, x):
        conv_d1, conv_d2, conv_d3, conv_d4 = self.encoder(x)
        conv_u2 = self.decoder([conv_d1, conv_d2, conv_d3, conv_d4])
        return self.classification(conv_u2)


class UNet3DMulti(UNet3D):
    def __init__(self, base_filters=32, n_class=5):
        super(UNet3DMulti, self).__init__(base_filters)
        self.classification = Classification3DUnetMulti(base_filters, n_class)
