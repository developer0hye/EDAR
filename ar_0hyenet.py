from torch import nn

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, ):
        super(EncoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class AR_0hyeNet(nn.Module):
    def __init__(self):
        super(AR_0hyeNet, self).__init__()

        self.encoder = EncoderBlock()

        self.nonlinear = self.make_layer(8)

        self.decoder = DecoderBlock()

    def forward(self, x):

        x = self.encoder(x)
        output = self.nonlinear(x)
        output = self.decoder(output + x)

        return output

    def make_layer(self, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)