import torch
import torch.nn as nn
from hw2_1_models import Generator
import random
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 1515
MODEL_PATH = "models_file/hw2_dcgan_gen.pt"
Z_DIM = 100
IMAGES_NUM = 32

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

G = Generator(100, 3, 64)
G.load_state_dict(torch.load(MODEL_PATH, map_location=device))

fixed_noise = torch.randn(IMAGES_NUM, Z_DIM, 1, 1).to(device)
generated_images = G(fixed_noise)

image_grid = torchvision.utils.make_grid(
    generated_images[:IMAGES_NUM], padding=2, normalize=True
)

plt.axis("off")
plt.title("DC-GAN")
plt.imshow(image_grid.cpu().permute(1, 2, 0))
plt.show()
