import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, params):
        super(Generator, self).__init__()

        # Model parameters
        mult = params.image_size // 8
        z_size = params.z_size
        G_h_size = params.G_h_size
        n_colors = params.n_colors

        main = torch.nn.Sequential()

        main.add_module('ConvTranspose2d', torch.nn.ConvTranspose2d(
            z_size, G_h_size * mult,
            kernel_size=4, stride=1, padding=0, bias=False)
        )
        main.add_module('BN', torch.nn.BatchNorm2d(G_h_size * mult))
        main.add_module('SELU', torch.nn.SELU(inplace=True))

        i = 1
        while mult > 1:
            main.add_module('ConvTranspose2d %d' % i, torch.nn.ConvTranspose2d(
                G_h_size * mult, G_h_size * (mult // 2),
                kernel_size=4, stride=2, padding=1, bias=False)
            )
            main.add_module('BN1 %d' % i, torch.nn.BatchNorm2d(G_h_size * (mult // 2)))
            main.add_module('SELU1 %d' % i, torch.nn.SELU(inplace=True))
            main.add_module('Conv2d %d' % i, torch.nn.Conv2d(
                G_h_size * (mult // 2), G_h_size * (mult // 2),
                kernel_size=3, padding=1, bias=False)
            )
            main.add_module('BN2 %d' % i, torch.nn.BatchNorm2d(G_h_size * (mult // 2)))
            main.add_module('SELU2 %d' % i, torch.nn.SELU(inplace=True))
            mult = mult // 2
            i += 1

        main.add_module('ConvTranspose2d_Out', torch.nn.ConvTranspose2d(
            G_h_size, G_h_size,
            kernel_size=4, stride=2, padding=1, bias=False)
        )
        main.add_module('Conv2d_Out', torch.nn.Conv2d(
            G_h_size, n_colors,
            kernel_size=3, padding=1, bias=False)
        )
        main.add_module('Tanh_Out', torch.nn.Tanh())
        self.main = main

    def forward(self, x):
        x = self.main(x)
        return x
