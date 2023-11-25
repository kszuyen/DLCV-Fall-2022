import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # print(embeddings.shape)
        return embeddings


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t):
        x = self.conv1(x)
        t = self.time_mlp(t)
        t = t.view(t.shape[0], t.shape[1], 1, 1)
        # t = t[(..., ) + (None, ) * 2]
        x = x + t
        x = self.conv2(x)
        return x


class DoubleConv_with_label(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch+1, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t):
        x = self.conv1(x)
        t = self.time_mlp(t)
        t = t[(..., ) + (None, ) * 2]
        x = x + t
        x = self.conv2(x)
        return x


class UNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, image_size=28, num_classes=10, features=[64, 128, 256, 512]):
        super().__init__()
        self.image_size = image_size
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        """  Lable Embedding  """
        self.label_embed = nn.Embedding(num_classes, image_size*image_size)

        """  Time Embedding  """
        self.time_dim = image_size*4
        self.time_embed = SinusoidalPositionEmbeddings(self.time_dim)

        """  Down  """
        self.downs.append(DoubleConv_with_label(
            in_ch, features[0], self.time_dim))
        in_ch = features[0]

        for feature in features[1:]:
            self.downs.append(DoubleConv(in_ch, feature, self.time_dim))
            in_ch = feature

        """  Up  """
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature, self.time_dim))

        self.bottleneck = DoubleConv(
            features[-1], features[-1]*2, self.time_dim)
        self.finalConv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x, t, labels):
        label_emb = self.label_embed(labels).view(
            labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, label_emb], dim=1)

        t_emb = self.time_embed(t)

        skip_connections = []
        for down in self.downs:
            x = down(x, t_emb)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x, t_emb)

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection:
                # height and width
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip, t_emb)  # double conv

        return self.finalConv(x)


def test():

    # x = torch.rand(64, 3, 28, 28)
    # time = torch.randint(0, 200, (64,)).long()
    # label = torch.randint(0, 10, (64,)).long()
    # print(x.shape)
    # print(time.shape)
    # print(label.shape)

    # model = UNET(3, 3, 28, num_classes=10)
    # preds = model(x, time, label)
    # assert preds.shape == x.shape
    # print('passed')

    # x = torch.rand(1, 3, 28, 28)
    # time = torch.randint(0, 200, (1,)).long()
    # label = torch.randint(0, 10, (1,)).long()
    # print(x.shape)
    # print(time.shape)
    # print(label.shape)

    # model = UNET(3, 3, 28, num_classes=10)
    # model.eval()
    # preds = model(x, time, label)
    # assert preds.shape == x.shape
    t = torch.full((64,), 40, dtype=torch.long)
    label = torch.randint(0, 10, (64,)).long()
    print(t)
    print(label)
    print(t.shape)
    print(label.shape)

    for i in range(1000):
        if i % 100 == 0:
            print(i)


if __name__ == '__main__':
    test()
