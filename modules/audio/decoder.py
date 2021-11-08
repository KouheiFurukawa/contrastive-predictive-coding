from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ConvTranspose1d(128, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 1, kernel_size=9, stride=5, padding=4, output_padding=4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)
