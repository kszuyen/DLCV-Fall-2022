import os
import imageio
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class hw2_1_dataset(Dataset):
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
        
class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()

# def normalize(T, new_min, new_max):
#     v_min, v_max = T.min(), T.max()
#     return (T - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


def gradient_penalty(critic, real, fake, device="cpu"):
    # batch_size, channels, h, w = real.shape
    # epsilon = torch.rand((batch_size, channels, h, w)
    #                      ).expand_as(real).to(device)
    epsilon = torch.randn((real.shape[0], 1, 1, 1)
                         ).expand_as(real).to(device)
    interpolated_images = (real * epsilon + fake * (1 - epsilon)).to(device)

    # calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(device),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    # gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=1)+1e-12)
    # gradient_norm = gradient.norm(2, dim=1)
    # gradient_penalty = torch.mean((gradient_norm-1) ** 2)
    gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def check_D_accuracy(G, D, z_dim, dataloader, device):
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for real_image in dataloader:
            real_image = real_image.to(device)
            noise = torch.randn((real_image.shape[0], z_dim, 1, 1)).to(device)
            fake_image = G(noise).to(device)

            scores = D(real_image).view(-1)
            # scores = nn.Sigmoid(scores)

            # real_label = torch.ones_like(scores).to(device)
            # _, predictions = scores.max(1)  # value, index

            num_correct += (scores > 0).sum().item()
            num_samples += scores.size(0)

            scores = D(fake_image).view(-1)
            # fake_label = torch.zeros_like(scores).to(device)
            # _, predictions = scores.max(1)  # value, index

            num_correct += (scores < 0).sum().item()
            num_samples += scores.size(0)

        print(
            f'D got {num_correct}/{num_samples}, with accuracy {float(num_correct)/float(num_samples):.2f}')

    return num_correct / num_samples
