import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import os, sys
import pandas as pd
import imageio.v2 as io
import json, csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
csv_dir = sys.argv[1]
image_folder = sys.argv[2]
output_csv = sys.argv[3]

# csv_dir = "/home/kszuyen/DLCV/hw4-kszuyen/hw4_data/office/val.csv"
# image_folder = "/home/kszuyen/DLCV/hw4-kszuyen/hw4_data/office/val"
# output_csv = "hw4_2_output.csv"
###

### inference with Setting C
model_dir = "C_wolabel_fullmodel.pt"
id2label_json_dir = "id2label.json"
with open(id2label_json_dir) as f:
    label_dict = json.load(f)

# load model
dropout = 0.1
resnet = models.resnet50(weights=None).to(device)
resnet.fc = nn.Sequential(
    nn.Linear(2048, 2048, bias=False),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(2048, 1000, bias=False),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1000, 1000, bias=False),
    nn.BatchNorm1d(1000),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1000, 65)
)
resnet = resnet.to(device)
resnet.eval()

resnet.load_state_dict(torch.load(model_dir, map_location=device))

# dataset
class SSL_Dataset(Dataset):
    # Dataset modified for inference
    def __init__(self, image_folder, csv_dir, json_file_dir=None, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_dir)
        self.image_dir = image_folder
        self.json_file_dir = json_file_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data.iloc[index, 0]
        image_name = self.data.iloc[index, 1]
        label = self.data.iloc[index, 2]

        image = io.imread(os.path.join(self.image_dir, image_name))

        if self.transform:
            image = self.transform(image.shape[:1])(image)

        if self.json_file_dir:
            with open(self.json_file_dir) as f:
                label_dict = json.load(f)
            label = list(label_dict.keys())[list(label_dict.values()).index(label)]
            label = torch.tensor(int(label))

        return image_id, image_name, image, label

def transform(img_shape):
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(min(img_shape)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return img_transform

test_dataset = SSL_Dataset(
    image_folder=image_folder,
    csv_dir=csv_dir,
    json_file_dir=id2label_json_dir,
    transform=transform,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False
)

# inference step
with open(output_csv, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["id", "filename", "label"])

    with torch.no_grad():
        for ids, image_names, images, labels in tqdm(test_loader):
            mini_batch_size = labels.shape[0]
            images, labels = images.to(device), labels.to(device)
            scores = resnet(images)

            _, predictions = scores.max(1)

            for i in range(mini_batch_size):
                writer.writerow([int(ids[i]), image_names[i], label_dict[str(int(predictions[i]))]])
