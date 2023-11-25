from torch.utils.data import Dataset
import pandas as pd
# import imageio.v2 as iio
import imageio as iio
from torchvision import transforms
import os


class hw2_3_dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

        img = iio.imread(img_path)
        label = self.annotations.iloc[index, 1]
        if self.transform:
            img = self.transform(img)

        return img, label


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
