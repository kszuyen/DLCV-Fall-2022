import torch
import torch.nn as nn
import pandas as pd
import os
import imageio.v2 as io
from torchvision import transforms

from torch.utils.data import Dataset
import json

class SSL_Dataset(Dataset):
    def __init__(self, root_dir, json_file_dir=None, data_type="train", transform=None):
        super().__init__()
        self.data = pd.read_csv(os.path.join(root_dir, f"{data_type}.csv"))
        self.image_dir = os.path.join(root_dir, data_type)
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

        return image_id, image, label

def img_transform(img_shape):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(min(img_shape)),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform


if __name__ == "__main__":
    dataset = SSL_Dataset(
        root_dir='/home/kszuyen/DLCV/hw4-kszuyen/hw4_data/office',
        # json_file_dir="id2label.json",
        transform=img_transform,
        data_type="train"
    )

    for i, (image_id, image, label) in enumerate(dataset):
        print(label)
        if i > 5:
            break
        
        