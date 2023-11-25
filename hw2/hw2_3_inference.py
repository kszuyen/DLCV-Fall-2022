import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import os
import imageio
from tqdm import tqdm
import csv

data_dir = sys.argv[1]
output_dir = sys.argv[2]
###
# data_dir = "hw2_data/digits/usps/data"
# output_dir = "usps.csv"
###
dataset_name = "svhn" if "svhn" in data_dir else "usps"
img_channels = 3 if dataset_name=="svhn" else 1
model_dir = f"dann_mnistm_2_{dataset_name}.pth"
    
batch_size = 128
image_size = 28
alpha = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class test_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        img_name = self.all_images[index].split('/')[-1]
        img = imageio.imread(os.path.join(
            self.root_dir, self.all_images[index]))
        if self.transform:
            img = self.transform(img)
        return img, img_name

def img_transform(img_channels=3, image_size=28):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * img_channels,
            std=[0.5] * img_channels
        )
    ])

model = CNNModel()
model.to(device)
checkpoint = torch.load(model_dir, map_location=device)
model.load_state_dict(checkpoint['dann_model'])
model.eval()

dataset = test_dataset(data_dir, transform=img_transform(img_channels=img_channels))
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

loop = tqdm(loader, total=len(loader))

with open(output_dir, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['image_name', 'label'])
    for (batch_img, batch_img_name) in loop:

        batch_size = len(batch_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)

        batch_img = batch_img.to(device)
        input_img = input_img.to(device)

        input_img.resize_as_(batch_img).copy_(batch_img)

        class_output, domain_output = model(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]

        for i in range(batch_size):
            writer.writerow([batch_img_name[i], pred[i].item()])






