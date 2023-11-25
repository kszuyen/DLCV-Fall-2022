import torch
import torch.nn as nn
# from hw2_1_models import Generator
import random
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
import sys
import os
# from face_recog import face_recog
# from pytorch_fid.fid_score import calculate_fid_given_paths
# from time import time
# start = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = sys.argv[1]
MODEL_DIR = "dcgan_checkpoint.pth"

# OUTPUT_DIR = "output"
# MODEL_DIR = "models_file/dcgan_checkpoint.pth"
# VALID_PATH = "hw2_data/face/val"
Z_DIM = 100

RANDOM_SEED = 160
# print(f"random seed: {RANDOM_SEED}")
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(nz, ngf*8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf*8, ngf*4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf*4, ngf*2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf*2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
                                         4, 2, 1, bias=True)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.tconv1(x)))
        x = F.leaky_relu(self.bn2(self.tconv2(x)))
        x = F.leaky_relu(self.bn3(self.tconv3(x)))
        x = F.leaky_relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
])

G = Generator(100, 3, 64)
checkpoint = torch.load(MODEL_DIR, map_location=device)
G.load_state_dict(checkpoint['Generator'])
G = G.to(device)
G.eval()

with torch.no_grad():
    test_noise = torch.randn(1000, Z_DIM, 1, 1).to(device)
    test_image = G(test_noise)
    test_image = reverse_transform(test_image)
    for i in range(1000):
        save_image(test_image[i], os.path.join(OUTPUT_DIR, str(i+1)+".png"))

# finish = time()
# print(f"Took {(finish-start)/60} sec")

# fr = face_recog(OUTPUT_DIR)
# print(f"Face recognition: {fr:.3f}")

# fid_score = calculate_fid_given_paths(
# (OUTPUT_DIR, VALID_PATH), 50, device, 2048, num_workers=4)
# print(f'fid_score: {fid_score}')
        
