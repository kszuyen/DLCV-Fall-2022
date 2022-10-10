import torch
import torch.nn as nn
import numpy as np
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import os
import albumentations as A
from torchvision.models import vgg16, VGG16_Weights

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
classes = ['Urban', 'Agriculture', 'Rangeland',
           'Forest', 'Water', 'Barren', 'Unknown']
colormap = {
    'Urban': [0, 1, 1],
    'Agriculture': [1, 1, 0],
    'Rangeland': [1, 0, 1],
    'Forest': [0, 1, 0],
    'Water': [0, 0, 1],
    'Barren': [1, 1, 1],
    'Unknown': [0, 0, 0]
}


augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.RandomResizedCrop(512, 512, scale=(0.7, 1.3), ratio=(0.8, 1.2)),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                       rotate_limit=35, p=0.5),

    # A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=1),
    A.OneOf([
        A.Compose([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                       b_shift_limit=15, p=0.75),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, p=1),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1),
            ], p=0.66),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1),
                A.GridDistortion(p=1)
            ], p=0.8),
        ])
    ], p=0.9),


    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class hw1_2_dataset(Dataset):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.transform = transform

    def __len__(self):
        # return int(len(os.listdir(self.rootdir))/2)
        c = 0
        for file in os.listdir(self.rootdir):
            if file.endswith('.jpg'):
                c += 1
        return c

    def __getitem__(self, idx):
        index = '{0:04}'.format(idx)
        img_path = index + "_sat.jpg"
        label_path = index + "_mask.png"

        image = imageio.imread(os.path.join(self.rootdir, img_path))
        mask = imageio.imread(os.path.join(self.rootdir, label_path))

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)

        mask = mask2label(mask)
        mask = transforms.ToTensor()(mask)

        return image, mask


def mask2label(mask):
    """ imput mask shape: (512, 512, 3), output label shape: (512, 512)"""

    mask = (mask >= 128).astype(int)  # original mask
    mask = 4 * mask[:, :, 0] + 2 * \
        mask[:, :, 1] + mask[:, :, 2]  # (512, 512) labeled

    class_mask = np.empty((512, 512))
    class_mask[mask == 3] = 0  # (Cyan: 011) Urban land
    class_mask[mask == 6] = 1  # (Yellow: 110) Agriculture land
    class_mask[mask == 5] = 2  # (Purple: 101) Rangeland
    class_mask[mask == 2] = 3  # (Green: 010) Forest land
    class_mask[mask == 1] = 4  # (Blue: 001) Water
    class_mask[mask == 7] = 5  # (White: 111) Barren land
    class_mask[mask == 0] = 6  # (Black: 000) Unknown
    return class_mask


def label2mask(pred):
    """ imput label shape: (512, 512), output mask shape: (512, 512, 3)"""

    masks_RGB = np.empty((512, 512, 3))
    masks_RGB[pred == 0] = [0, 255, 255]
    masks_RGB[pred == 1] = [255, 255, 0]
    masks_RGB[pred == 2] = [255, 0, 255]
    masks_RGB[pred == 3] = [0, 255, 0]
    masks_RGB[pred == 4] = [0, 0, 255]
    masks_RGB[pred == 5] = [255, 255, 255]
    masks_RGB[pred == 6] = [0, 0, 0]
    masks_RGB = masks_RGB.astype(np.uint8)

    return masks_RGB


def save_pred(pred_file, pred, image_index):
    imageio.imsave(os.path.join(pred_file, image_index+'.png'), pred)


class hw1_2_test_dataset(Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def __len__(self):
        # return int(len(os.listdir(self.rootdir))/2)
        c = 0
        for file in os.listdir(self.rootdir):
            if file.endswith('.jpg'):
                c += 1
        return c

    def __getitem__(self, idx):
        index = '{0:04}'.format(idx)
        img_path = index + '_sat.jpg'

        image = imageio.imread(os.path.join(self.rootdir, img_path))
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)

        return image
