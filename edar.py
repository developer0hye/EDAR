import common

import torch
import torch.nn as nn

class EDAR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDAR, self).__init__()

        n_resblock = 8
        n_feats = 64
        kernel_size = 3

        #DIV 2K mean
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_mean, rgb_std)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, 3, kernel_size)
        ]

        self.add_mean = common.MeanShift(rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return torch.clamp(x,0.0,1.0)