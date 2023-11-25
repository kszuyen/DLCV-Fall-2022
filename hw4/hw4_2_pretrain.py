import torch
from byol_pytorch import BYOL
from torchvision import models
from hw4_2_dataset import SSL_Dataset, img_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_epoch = 250
batch_size = 512
learning_rate = 3e-4

model_dir = "pretrained_model_SSL_backbone.pt"

resnet = models.resnet50(weights=None).to(device)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=learning_rate)

mini_dataset = SSL_Dataset(
    root_dir='/home/kszuyen/DLCV/hw4-kszuyen/hw4_data/mini',
    transform=img_transform,
    data_type="train"
)
mini_loader = DataLoader(
    dataset=mini_dataset, 
    batch_size=batch_size,
    shuffle=True
)

for epoch in range(num_epoch):
    loop = tqdm(mini_loader)
    loop.set_description(f"Epoch: {epoch}")
    for images_id, images, labels in loop: 
        images = images.to(device)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        loop.set_postfix({
            "loss": f"{loss:.4f}"
        })
    # save your improved network
    torch.save(resnet.state_dict(), model_dir)
        

