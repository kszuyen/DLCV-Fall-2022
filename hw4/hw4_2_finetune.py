import torch
import torch.nn as nn
from torchvision import models
from hw4_2_dataset import SSL_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTINGS = "E"

# saved model settings
LOAD_MODEL = False
saved_epoch = 33
saved_acc = 0.2906

# hyperparameters
num_epoch = 80
batch_size = 64
learning_rate = 3e-4
dropout = 0.1

json_file_dir = "id2label.json"
resnet_backbone_dir = [
    "hw4_data/pretrain_model_SL.pt", # with label
    "pretrained_model_SSL_backbone.pt", # without label
]
finetune_model_dir = {
    "A": "models_file/A_fullmodel.pt",
    "B": "models_file/B_withlabel_fullmodel.pt",
    "C": "models_file/C_wolabel_fullmodel.pt",
    "D": "models_file/D_withlabel_freezebackbone.pt",
    "E": "models_file/E_wolabel_freezebackbone.pt",
}

if __name__ == "__main__":
    if SETTINGS=="A":
        backbone_id, freeze_weights = None, False
    elif SETTINGS=="B":
        backbone_id, freeze_weights = 0, False
    elif SETTINGS=="C":
        backbone_id, freeze_weights = 1, False
    elif SETTINGS=="D":
        backbone_id, freeze_weights = 0, True
    elif SETTINGS=="E":
        backbone_id, freeze_weights = 1, True
    else:
        print("Settings error!\nPlease select one of {A, B, C, D, E}")
        sys.exit()

    print(f"Loading model with settings {SETTINGS}...")
    # load and model settings
    resnet = models.resnet50(weights=None).to(device)
    if backbone_id:
        resnet.load_state_dict(torch.load(resnet_backbone_dir[backbone_id], map_location=device))
    if freeze_weights:
        for param in resnet.parameters():
            param.requires_grad = False

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

    # optimizer and loss function
    optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # data augmentation
    def train_transform(img_shape):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(img_shape)),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.3),
            transforms.RandomResizedCrop(
                (128, 128), scale=(3/4, 4/3), ratio=(0.8, 1.2)),
            transforms.RandomAffine(
                    degrees=40, translate=(0.1, 0.2), scale=(0.8, 1.2)),
            ]), p=0.5),
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform
    def val_transform(img_shape):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(min(img_shape)),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform

    # dataset and dataloader
    train_dataset = SSL_Dataset(
        root_dir="hw4_data/office",
        json_file_dir=json_file_dir,
        data_type="train",
        transform=train_transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataset = SSL_Dataset(
        root_dir="hw4_data/office",
        json_file_dir=json_file_dir,
        data_type="val",
        transform=val_transform,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # load checkpoint
    if LOAD_MODEL:
        print("loading checkpoint...")
        resnet.load_state_dict(torch.load(finetune_model_dir[SETTINGS], map_location=device))
        best_acc = saved_acc
    else:
        saved_epoch = 0
        best_acc = 0

    # Training Start
    print("Training start.")
    for epoch in range(saved_epoch+1, num_epoch+1):

        """  training phase  """
        resnet.train()
        loop = tqdm(train_loader)
        loop.set_description(f"Train: {epoch}")
        for ids, images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            scores = resnet(images)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix_str(f"loss: {loss:.4f}")

        """  validation phase  """
        resnet.eval()
        num_correct = 0
        loop = tqdm(val_loader)
        loop.set_description(f"Validation:")
        for ids, images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            scores = resnet(images)
            _, predictions = scores.max(1)  # value, index

            num_correct += (predictions == labels).sum()
        
        acc = num_correct / len(val_dataset)
        print(f"validation set accuracy: {acc:.4f}")

        """  save model  """
        if acc > best_acc:
            best_acc = acc
            torch.save(resnet.state_dict(), finetune_model_dir[SETTINGS])
            print("model saved!")







