import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

# def weights_init(w):
#     """
#     Initializes the weights of the layer, w.
#     """
#     classname = w.__class__.__name__
#     if classname.find('conv') != -1:
#         nn.init.normal_(w.weight.data, 0.0, 0.02)
#     elif classname.find('bn') != -1:
#         nn.init.normal_(w.weight.data, 1.0, 0.02)
#         nn.init.constant_(w.bias.data, 0)

# Define the Generator Network


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(nz, ngf*8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf*8, ngf*4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf*4, ngf*2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf*2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
                                         4, 2, 1, bias=True)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.tconv1(x)))
        x = F.leaky_relu(self.bn2(self.tconv2(x)))
        x = F.leaky_relu(self.bn3(self.tconv3(x)))
        x = F.leaky_relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x

# Define the Discriminator Network


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf,
                               4, 2, 1, bias=True)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf*2,
                               4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf*2, ndf*4,
                               4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf*4, ndf*8,
                               4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.conv1(x), 0.2, True))
        x = self.dropout(F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True))
        x = self.dropout(F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True))
        x = self.dropout(F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True))

        x = torch.sigmoid(self.conv5(x))

        return x


class Critic(nn.Module):
    def __init__(self, n_channels, features_d):
        super().__init__()
        self.main = nn.Sequential(
            # input is (c) x 64 x 64
            nn.Conv2d(n_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (df) x 32 x 32
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (df*2) x 16 x 16
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (df*4) x 8 x 8
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (df*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# def initialize_weights(model):
#     for m in model.modules():
#         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
#             nn.init.normal_(m.weight.data, 0.0, 0.02)


def initialize_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)
    if isinstance(model, nn.BatchNorm2d):
        torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(model.bias, val=0)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    # x = torch.rand((N, in_channels, H, W))
    D = Discriminator(in_channels, ndf=64)
    # print('D')
    # initialize_weights(D)
    # assert D(x).shape == (N, 1, 1, 1)
    G = Generator(z_dim, in_channels, ngf=64)
    # print('G')
    # initialize_weights(G)
    # z = torch.randn((N, z_dim, 1, 1))
    # assert G(z).shape == (N, in_channels, H, W)
    # print('Success')
    # print(G.__class__.__name__.find('Conv'))
    # print(G.__class__.__name__.find('BatchNorm'))
    # print('Discriminator')
    # summary(D.cuda(), (in_channels, H, W))
    # print('Generator')
    # summary(G.cuda(), (z_dim, 1, 1))

if __name__ == '__main__':
    test()
