import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import DenseNet121_Weights


class hw1_1_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024*2*2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 50)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = self.conv7(x)
        x = x.view(-1, 1024*2*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class hw1_1_densenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT)
        # self.backbone = self.pretrained_densenet
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 50)
        )

    def forward(self, x):
        x = self.densenet(x)
        return x

def test():
    x = torch.rand(4, 3, 224, 224)
    # for i, k in enumerate(list(models.densenet121().children())):
    #     print(i, ': ', k)
    print(models.densenet121())
    model = hw1_1_densenet()
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()