import torch.nn as nn
import numpy as np
from torch.nn import Linear, Flatten, ReLU, Conv2d, Sequential, Tanh, Sigmoid, AdaptiveAvgPool2d

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    modules = []
    for module in args:
        if isinstance(module, Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return Sequential(*modules)


class ResBlock(nn.Module):
    def __init__(self, in_out_c=64):
        super(ResBlock, self).__init__()
        
        self.res = nn.Sequential(
            Conv2d(in_out_c, in_out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ReLU(inplace=True),
            Conv2d(in_out_c, in_out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        )

    def forward(self, x):
        res = self.res(x)
        return x + res


class QFAttention(nn.Module):
    def __init__(self, in_out_c=64):
        super(QFAttention, self).__init__()

        self.res = nn.Sequential(
            Conv2d(in_out_c, in_out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            ReLU(inplace=True),
            Conv2d(in_out_c, in_out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        )

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res(x) + beta
        return x + res


class FBCNN(nn.Module):
    def __init__(self, in_out_channels=3, nc=[64, 128, 256, 512], nb=4):
        def downsample(in_c, out_c):
            return Conv2d(in_c, out_c, kernel_size=2, stride=2, padding=0, dilation=1, groups=1, bias=True)
        
        def upsample(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
        super(FBCNN, self).__init__()
        self.nb=nb

        self.m_head = Conv2d(in_out_channels, nc[0], kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.m_down1 = sequential(*[ResBlock(nc[0]) for _ in range(nb)], downsample(nc[0], nc[1]))
        self.m_down2 = sequential(*[ResBlock(nc[1]) for _ in range(nb)], downsample(nc[1], nc[2]))
        self.m_down3 = sequential(*[ResBlock(nc[2]) for _ in range(nb)], downsample(nc[2], nc[3]))

        self.m_body_encoder = sequential(*[ResBlock(nc[3]) for _ in range(nb)])
        self.m_body_decoder = sequential(*[ResBlock(nc[3]) for _ in range(nb)])

        self.m_up3 = nn.ModuleList([upsample(nc[3], nc[2]), *[QFAttention(nc[2]) for _ in range(nb)]])
        self.m_up2 = nn.ModuleList([upsample(nc[2], nc[1]), *[QFAttention(nc[1]) for _ in range(nb)]])
        self.m_up1 = nn.ModuleList([upsample(nc[1], nc[0]), *[QFAttention(nc[0]) for _ in range(nb)]])

        self.m_tail = Conv2d(nc[0], in_out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)


        self.qf_pred = sequential(
            *[ResBlock(nc[3]) for _ in range(nb)],
            AdaptiveAvgPool2d((1,1)), Flatten(), Linear(512, 512), ReLU(), Linear(512, 512),
            ReLU(),
            Linear(512, 1),
            Sigmoid()
        )

        self.qf_embed = sequential(Linear(1, 512), ReLU(), Linear(512, 512), ReLU(), Linear(512, 512), ReLU())
        
        self.to_gamma_3 = sequential(Linear(512, nc[2]), Sigmoid())
        self.to_beta_3 =  sequential(Linear(512, nc[2]), Tanh())
        self.to_gamma_2 = sequential(Linear(512, nc[1]), Sigmoid())
        self.to_beta_2 =  sequential(Linear(512, nc[1]), Tanh())
        self.to_gamma_1 = sequential(Linear(512, nc[0]), Sigmoid())
        self.to_beta_1 =  sequential(Linear(512, nc[0]), Tanh())


    def forward(self, x, qf_input=None):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        x = self.m_body_decoder(x)
        qf_embedding = self.qf_embed(qf_input) if qf_input is not None else self.qf_embed(self.qf_pred(x))
        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)

        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)

        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)


        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i+1](x, gamma_3,beta_3)

        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i+1](x, gamma_2, beta_2)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i+1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x