import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from hw2_1_models import Generator, Discriminator, initialize_weights
from hw2_1_utils import hw2_1_dataset
from face_recog import face_recog
import numpy as np
import random
from pytorch_fid.fid_score import calculate_fid_given_paths

# import imageio
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = True

# Hyperparameters
# RANDOM_SEED = 369
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.5
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 300
FEATURES_DISC = 64
FEATURES_GEN = 64  # to match 1024 from the paper (64x16 = 1024)

# COLAB_PATH = "/content/drive/MyDrive/hw2-kszuyen"
TRAIN_PATH = "hw2_data/face/train"
VALID_PATH = "hw2_data/face/val"
path_dict = {
    'OUTPUT_DIR': "DCGAN_results/output",
    'MODEL_CHECKPOINT': 'models_file/dcgan_checkpoint.pt',
    'STEPGRID': "DCGAN_results/grid",
}


def main():
    # print("Random Seed: ", RANDOM_SEED)
    # random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)

    # if TRAINING_IN_COLAB:
    #     for path in path_dict:
    #         path_dict[path] = os.path.join(COLAB_PATH, path_dict[path])
        # os.chdir(COLAB_PATH)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.5, 0.5, 0.5),
        #     (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        # transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        # transforms.Lambda(lambda t: t * 255.),
        # transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        # transforms.ToPILImage(),
    ])

    dataset = hw2_1_dataset(
        root_dir=TRAIN_PATH, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
    D = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
    optimizer_G = optim.Adam(
        G.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(
        D.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))

    if LOAD_MODEL:
        print("Loading Checkpoint...")
        checkpoint = torch.load(
            path_dict['MODEL_CHECKPOINT'], map_location=device)
        G.load_state_dict(checkpoint['Generator'])
        D.load_state_dict(checkpoint['Discriminator'])
        # optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        # optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        best_face_recog = checkpoint['best_face_recog']
        best_fid_score = checkpoint['best_fid_score']
        saved_epoch = checkpoint['epoch']
        print(f"Best face recognition: {best_face_recog:.4f}")
        print(f"Best fid score: {best_fid_score:.4f}")
        print("~~~Finished~~~")
    else:
        G.apply(initialize_weights)
        D.apply(initialize_weights)
        best_face_recog = 0.0
        best_fid_score = 1e6
        saved_epoch = 0
        print(f"Set best_face_recog: {best_face_recog}, best_fid_score: {best_fid_score}")

    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    # criterion = label_smooth_loss(2)

    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)
    # step = 0
    G.train()
    D.train()

    for epoch in range(saved_epoch+1, NUM_EPOCHS):

        real_correct = 0
        fake_correct = 0
        num_samples = 1e-16
        # output_image_count = 0

        loop = tqdm(loader, total=len(loader))
        loop.set_description(f'Epoch: {epoch}')

        for batch_idx, real_image in enumerate(loop):
            """ Training Phase """
            G.train()
            D.train()
            mini_batch_size = real_image.shape[0]
            real_image = real_image.to(device)
            noise = torch.randn((mini_batch_size, Z_DIM, 1, 1), device=device)
            fake_image = G(noise)

            """  Train discriminator max log(D(x)) + log(1 - D(G(z)))  """
            D.zero_grad()
            D_real = D(real_image).view(-1)
            D_fake = D(fake_image.detach()).view(-1)
            # label = torch.full((real_image.shape[0], ), 1., device=device)
            rand_real_label = ((1.2 - 0.7) * torch.rand(mini_batch_size)  + 0.7).view(D_real.shape).to(device)
            rand_fake_label = ((0.3) * torch.rand(mini_batch_size)).view(D_fake.shape).to(device)
            # flip labels:
            flip_count = 0
            for i in range(mini_batch_size):
                if np.random.choice([0, 1], p=[0.98, 0.02]):
                    rand_real_label[i], rand_fake_label[i] = rand_fake_label[i], rand_real_label[i]
                    flip_count += 1

            # loss_D_real = criterion(D_real, label)
            # loss_D_real.backward()
            loss_D_real = criterion(D_real, rand_real_label)
            # label.fill_(0.)
            # loss_D_fake = criterion(D_fake, label)
            # loss_D_fake.backward()
            loss_D_fake = criterion(D_fake, rand_fake_label) 

            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            
            """  check D acc  """
            with torch.no_grad():
                real_label = torch.ones_like(D_real).to(device)
                # _, predictions = D_real.max(1)  # value, index
                D_real_pred = D_real > 0.5
                real_correct += (D_real_pred == real_label).sum().item()
                # num_samples += real_label.size(0)

                # fake_label = torch.zeros_like(D_fake).to(device)
                # _, predictions = D_fake.max(1)  # value, index
                D_fake_pred = D_fake < 0.5
                fake_correct += (D_fake_pred == real_label).sum().item()

                num_samples += real_label.size(0)
                D_real_acc = real_correct / num_samples
                D_fake_acc = fake_correct / num_samples
            # if (D_real_acc + D_fake_acc) < 1.8:
            optimizer_D.step()


            
            """  Train Generator min log(1-D(G(z))) <--> max log(D(G(z)))  """
            G.zero_grad()
            output = D(fake_image).view(-1)
            loss_G = criterion(output, torch.ones_like(output).to(device))
            loss_G.backward()
            # if D_real_acc > 0.6 and D_fake_acc > 0.6:
            optimizer_G.step()

            loop.set_postfix({
                'loss_D': f'{loss_D.item():.4f}', 
                'loss_G': f'{loss_G.item():4f}',
                'D_real_acc': f'{D_real_acc:.3f}',
                'D_fake_acc': f'{D_fake_acc:.3f}',
                'D_real_mean': f'{torch.mean(D_real).item():.3f}',
                'D_fake_mean': f'{torch.mean(D_fake).item():.3f}',
                'flip_count': f'{flip_count}/{mini_batch_size}'
                })

        """  evaluation phase  """
        G.eval()
        D.eval()

        # evaluate with face recognition score
        with torch.no_grad():
            # check_D_accuracy(G, C, loader)

            eval_image = G(fixed_noise)
            eval_image = reverse_transform(eval_image)
            image_grid = torchvision.utils.make_grid(
                eval_image, padding=2, normalize=True
            )
            save_image(image_grid, os.path.join(
                path_dict['STEPGRID'], str(epoch)+".png"))
            test_noise = torch.randn(1000, Z_DIM, 1, 1).to(device)
            test_image = G(test_noise)
            test_image = reverse_transform(test_image)
            for i in range(1000):
                save_image(test_image[i], os.path.join(
                    path_dict['OUTPUT_DIR'], str(i+1)+".png"))

        fr = face_recog(path_dict['OUTPUT_DIR'])
        print(f"Face recognition: {fr:.3f}")
        if fr > best_face_recog:
            best_face_recog = fr

        fid_score = calculate_fid_given_paths(
            (path_dict['OUTPUT_DIR'], VALID_PATH), 50, device, 2048, num_workers=4)
        print(f'fid_score: {fid_score}')
        if fid_score < best_fid_score:
            best_fid_score = fid_score
            torch.save({'Generator': G.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'Discriminator': D.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'epoch': epoch,
                        'best_fid_score': best_fid_score,
                        'best_face_recog': best_face_recog,

                        }, path_dict['MODEL_CHECKPOINT'])
            print(
                f"~~~~Model saved at fid score: {best_fid_score:.3f}, face recognition: {fr:.3f}~~~~\n")


if __name__ == "__main__":
    main()
    