import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import torch
from torchvision import transforms
from PIL import Image
import random
import json
import os
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, json_dir, transform=None, mode="TRAIN"):
        """
        :param data_folder: folder where data files are stored - /Users/skye/docs/image_dataset/dataset
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder = data_folder

        with open(json_dir) as f:
            self.json_file = json.load(f)

        self.images = self.json_file["images"]
        self.annotations = self.json_file["annotations"]

        self.dataset_size = len(self.images)
        self.captions = {}
        # initialize dict
        for img in self.images:
            self.captions[img["id"]] = []
        for cap in self.annotations:
            self.captions[cap["image_id"]].append(cap["caption"])
    
        self.transform = transform
        self.mode = mode

    def __getitem__(self, i):
        
        
        img = Image.open(os.path.join(self.data_folder, self.images[i]["file_name"])).convert("RGB")
        h, w = img.size
        if self.transform:
            img = self.transform(h, w)(img)

        if self.mode == 'TRAIN':
            caption = random.choice(self.captions[self.images[i]["id"]])
            tokenizer = Tokenizer.from_file("hw3_data/caption_tokenizer.json")
            tokenized_caption = tokenizer.encode(caption).ids

            return img, torch.tensor(tokenized_caption)
        
        else:
            # For validation ofrtesting, also return all captions to find score
            img_name = self.images[i]["file_name"].split(".")[0]
            all_captions = self.captions[self.images[i]["id"]]
            return img, str(img_name), all_captions

    def __len__(self):
        return self.dataset_size

def img_transform(h, w):
    return transforms.Compose([
        transforms.CenterCrop((min(h, w))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
class MyCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def get_loader(image_folder, json_dir, img_transform, mode="TRAIN", batch_size=32, pad_idx=0):
    dataset = CaptionDataset(image_folder, json_dir,transform=img_transform, mode=mode)

    if mode=="TRAIN":
        loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle = True,
            collate_fn = MyCollate(pad_idx=pad_idx)
            )
    else:
        loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle = False,
            )
    return loader

if __name__ == "__main__":
    image_dir = "hw3_data/p2_data/images/train"
    json_dir = "hw3_data/p2_data/train.json"
    # dataset = CaptionDataset(image_dir, json_dir,transform=img_transform, mode="TRAIN")
    # print()
    # print(dataset[0])
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.imshow(np.transpose(dataset[12][0].numpy(), (1,2,0)))
    # plt.savefig("test.png")
    # print(tokenize_and_pad("this caption."))
    # loader = get_loader(image_folder=image_dir, json_dir=json_dir, img_transform=img_transform)
    with open(json_dir) as f:
        json_file = json.load(f)

    images = json_file["images"]
    annotations = json_file["annotations"]

    captions = {}
    # initialize dict
    for img in images:
        captions[img["id"]] = []
    for cap in annotations:
        captions[cap["image_id"]].append(cap["caption"])
    # print(captions)
    print(captions[images[0]["id"]])
    dataset = CaptionDataset(image_dir, json_dir,transform=img_transform, mode="VAL")
    for img, img_name, all_cap in dataset:
        print(img, img_name, all_cap)
        break
    val_loader = get_loader(image_folder=image_dir, json_dir=json_dir, img_transform=img_transform, mode="VAL", batch_size=1, pad_idx=0)

    for img, img_name, all_cap in val_loader:
        print(img, img_name, all_cap)
        break
    # for img, cap in loader:
    #     print(img.shape)
    #     print(cap.shape)
    #     # plt.imshow(np.transpose(img[0].numpy(), (1,2,0)))
    #     # plt.savefig("test.png")
    #     print(cap[0])
    #     tokenizer = Tokenizer.from_file("hw3_data/caption_tokenizer.json")
    #     print(tokenizer.decode(cap[0].numpy()))

    #     break