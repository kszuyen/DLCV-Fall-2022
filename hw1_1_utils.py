import os
import imageio
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset


# dataset with labels
class hw1_1_dataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            y_label = int(index / 450)
            image_id = index % 450
        else:
            y_label = int(index / 50)
            image_id = (index % 50) + 450
        img_path = str(y_label) + '_' + str(image_id) + '.png'
        img = imageio.imread(os.path.join(self.root_dir, img_path))

        if self.transform:
            img = self.transform(img)
        y_label = torch.tensor(y_label)
        return (img, y_label)

# testing dataset without labels


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


# Normalize to mean, std
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
# apply data augmentation methods


def transform(image_size=32):
    return {
        'train':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(brightness=0.3),
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(3/4, 4/3), ratio=(0.8, 1.2)),
                transforms.RandomAffine(
                    degrees=40, translate=(0.1, 0.2), scale=(0.8, 1.2)),
            ]), p=0.5),
            # transforms.RandomRotation(degrees=30)
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    }
