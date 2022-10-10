from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import sys
from torchvision.models import vgg16, VGG16_Weights
import os
import imageio
import numpy as np
# from mean_iou_evaluate import read_masks, mean_iou_score
# import time

# start = time.time()
# input_dir = 'hw1_data/p2_data/validation'
# output_dir = 'hw1_2_outputfile'
input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_dir = './vgg16_fcn8.pt'

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class vgg16_fcn8(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        feats = list(vgg16(weights=VGG16_Weights.DEFAULT).features.children())

        self.feat1 = nn.Sequential(*feats[0:5])
        self.feat2 = nn.Sequential(*feats[5:10])
        self.feat3 = nn.Sequential(*feats[10:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])
        self.feat6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
        )

        self.upsamplex64 = nn.ConvTranspose2d(
            1024, num_classes, kernel_size=64, stride=64)
        self.upsamplex32 = nn.ConvTranspose2d(
            512, num_classes, kernel_size=32, stride=32)
        self.upsamplex16 = nn.ConvTranspose2d(
            512, num_classes, kernel_size=16, stride=16)
        self.upsamplex8 = nn.ConvTranspose2d(
            256, num_classes, kernel_size=8, stride=8)
        self.upsamplex4 = nn.ConvTranspose2d(
            128, num_classes, kernel_size=4, stride=4)
        self.upsamplex2 = nn.ConvTranspose2d(
            64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        # 3*512*512
        feat1 = self.feat1(x)
        # 64*256*256
        feat2 = self.feat2(feat1)
        # 128*128*128
        feat3 = self.feat3(feat2)
        # 256*64*64
        feat4 = self.feat4(feat3)
        # 512*32*32
        feat5 = self.feat5(feat4)
        # 512*16*16
        feat6 = self.feat6(feat5)
        # 1024*8*8
        score = self.upsamplex64(feat6) + self.upsamplex32(feat5) + \
            self.upsamplex16(feat4) + self.upsamplex8(feat3) + \
            self.upsamplex4(feat2) + self.upsamplex2(feat1)

        return score


class hw1_2_test_dataset(Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.image_list = [file for file in os.listdir(
            self.rootdir) if file.endswith('.jpg')]

    def __len__(self):
        # return int(len(os.listdir(self.rootdir))/2)
        return len(self.image_list)

    def __getitem__(self, idx):
        index = '{0:04}'.format(idx)
        img_path = index + '.jpg'

        image = imageio.imread(os.path.join(self.rootdir, img_path))
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)

        return image


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


# load model
model = vgg16_fcn8()
model.load_state_dict(torch.load(model_dir, map_location=device))
model.to(device)
model.eval()

# load testing dataset
test_dataset = hw1_2_test_dataset(input_dir)
loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

# output result predict masks
with torch.no_grad():
    image_count = 0
    for batch_index, img in enumerate(loader):
        img.to(device)

        pred = model(img)
        _, predicted = torch.max(pred.data, 1)
        predicted = predicted.detach().cpu().numpy()

        for i in range(len(predicted)):
            pred_mask = label2mask(predicted[i, :, :])
            image_num = "%04d" % image_count
            image_count += 1
            imageio.imsave(os.path.join(
                output_dir, image_num+'.png'), pred_mask)

# finish = time.time()

# print('Finished in ', (finish-start)/60, 'min')
# ground_truth = read_masks(input_dir)
# test_predict = read_masks(output_dir)
# mean_iou_score(test_predict, ground_truth)
