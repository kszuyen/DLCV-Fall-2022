from torch.utils.data import DataLoader, Dataset
import os
import torch
import imageio.v2 as imageio
import torch.nn as nn
import csv
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
import sys
# import time
# start = time.time()

# input_dir = 'hw1_data/p1_data/val_50'
# output_dir = 'test.csv'
input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_dir = './hw1_1_densenet.pt'


class hw1_1_densenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT)
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


class hw1_1_test_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        img = imageio.imread(os.path.join(
            self.root_dir, self.all_images[index]))
        if self.transform:
            img = self.transform(img)
        return img

    def getimagetitle(self, index):
        return self.all_images[index]


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = hw1_1_densenet()
model.load_state_dict(torch.load(
    model_dir, map_location=device))
model.to(device)
model.eval()

# load testing dataset
test_dataset = hw1_1_test_dataset(
    input_dir,
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]))
loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# output result into csv
with open(output_dir, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['filename', 'label'])
    with torch.no_grad():
        for i, img in enumerate(loader):
            img = img.to(device)
            scores = model(img)
            _, predictions = scores.max(1)  # value, index

            writer.writerow([test_dataset.getimagetitle(i), int(predictions)])
