import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from hw2_1_models import Generator, Discriminator, initialize_weights, weights_init, Critic
from hw2_1_utils import hw2_1_dataset, gradient_penalty
from face_recog import face_recog
import random
from pytorch_fid.fid_score import calculate_fid_given_paths

# import imageio
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_IN_COLAB = True
LOAD_MODEL = True

# Hyperparameters
RANDOM_SEED = 168
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-3
BETA1 = 0.5
BETA2 = 0.999
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 300
FEATURES_DISC = 64
FEATURES_GEN = 64  # to match 1024 from the paper (64x16 = 1024)
CRITIC_ITERATIONS = 10
LAMBDA_GP = 8


COLAB_PATH = "/content/drive/MyDrive/hw2-kszuyen"
TRAIN_PATH = "hw2_data/face/train"
VALID_PATH = "hw2_data/face/val"
path_dict = {
    # 'TRAIN_PATH': "hw2_data/face/train",
    # 'VALID_PATH': "hw2_data/face/val",
    # 'OUTPUT_DIR': "dcgan_output_image",
    'OUTPUT_DIR': "output_image",
    # 'GEN_DIR': "models_file/hw2_dcgan_gen.pt",
    # 'DISC_DIR': "models_file/hw2_dcgan_disc.pt"
    # 'CRITIC_DIR': "models_file/wgangp_checkpoint.pt",
    'MODEL_CHECKPOINT': 'models_file/wgangp_checkpoint_v2.pt',
    'STEPGRID': "wgangp_grid_v2",
}

def main():
    print("Random Seed: ", RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if TRAINING_IN_COLAB:
        for path in path_dict:
            path_dict[path] = os.path.join(COLAB_PATH, path_dict[path])
        # os.chdir(COLAB_PATH)

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))
    ])

    dataset = hw2_1_dataset(
        root_dir=TRAIN_PATH, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
    # D = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
    D = Critic(IMAGE_CHANNELS, FEATURES_DISC).to(device)
    optimizer_G = optim.Adam(
        G.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(
        D.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
    # wgangp_checkpoint = torch.load(
    #         path_dict['CRITIC_DIR'], map_location=device)
    if LOAD_MODEL:
        print("Loading Checkpoint...")
        checkpoint = torch.load(
            path_dict['MODEL_CHECKPOINT'], map_location=device)
        G.load_state_dict(checkpoint['Generator'])
        D.load_state_dict(checkpoint['Discriminator'])
        # D.load_state_dict(wgangp_checkpoint['Critic'])
        # optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        # optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        best_face_recog = checkpoint['best_face_recog']
        best_fid_score = checkpoint['best_fid_score']
        saved_epoch = checkpoint['epoch']
        print(f"Best face recognition: {best_face_recog:.4f}")
        print(f"Best fid score: {best_fid_score:.4f}")
        print("~~~Finished~~~")
    else:
        G.apply(weights_init)
        D.apply(weights_init)
        best_face_recog = 0.0
        best_fid_score = 1e6
        saved_epoch = 0

    # criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
    # criterion = label_smooth_loss(2)

    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)

    # writer_real = SummaryWriter(f"logs/real")
    # writer_fake = SummaryWriter(f"logs/fake")
    # step = 0

    G.train()
    D.train()

    for epoch in range(saved_epoch+1, NUM_EPOCHS):

        real_correct = 0
        fake_correct = 0
        num_correct = 0
        num_samples = 1e-16
        output_image_count = 0

        loop = tqdm(loader, total=len(loader))
        loop.set_description(f'Epoch: {epoch}')

        for batch_idx, real_image in enumerate(loop):
            """ Training Phase """
            G.train()
            D.train()

            real_image = real_image.to(device)
            # noise = torch.randn((real_image.shape[0], Z_DIM, 1, 1), device=device)

            # """  Train discriminator max log(D(x)) + log(1 - D(G(z)))  """
            """  Training Critic  """
            for it in range(CRITIC_ITERATIONS):
                noise = torch.randn(
                    (real_image.shape[0], Z_DIM, 1, 1)).to(device)
                fake_image = G(noise)
                C_real = D(real_image).view(-1)
                C_fake = D(fake_image).view(-1)
                gp = gradient_penalty(D, real_image, fake_image, device=device)

                loss_C = (
                    -(torch.mean(C_real) - torch.mean(C_fake)) + LAMBDA_GP * gp
                )
                D.zero_grad()
                loss_C.backward(retain_graph=True)
                optimizer_D.step()
                
                if it == (CRITIC_ITERATIONS-1):
                    mean = (torch.mean(C_real) - torch.mean(C_fake))/2
                    num_correct += (C_real > mean).sum().item()
                    num_samples += C_real.size(0)
                    num_correct += (C_fake < mean).sum().item()
                    num_samples += C_fake.size(0)
                C_acc = num_correct / num_samples
                # for p in C.parameters():
                #     p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            """  Training Generator: min -E(Critic(gen_fake))"""
            output = D(fake_image).view(-1)
            loss_G = -torch.mean(output)
            G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            loop.set_postfix({
                'real_score': torch.mean(C_real).item(),
                'fake_score': torch.mean(C_fake).item(),
                'C_acc': f'{C_acc:.2f}',
            })
            
            index = 0
            while output_image_count < 1000 and index < fake_image.shape[0]:
                save_image(fake_image[index], os.path.join(
                    path_dict['OUTPUT_DIR'], str(output_image_count)+".png"))
                output_image_count += 1
                index += 1

        """  evaluation phase  """
        G.eval()
        D.eval()

        # evaluate with face recognition score
        with torch.no_grad():
            # check_D_accuracy(G, C, loader)

            eval_image = G(fixed_noise)
            image_grid = torchvision.utils.make_grid(
                eval_image[:64], padding=2, normalize=True
            )
            save_image(image_grid, os.path.join(
                path_dict['STEPGRID'], str(epoch)+".png"))

        save_model = False
        fr = face_recog(path_dict['OUTPUT_DIR'])
        print(f"Face recognition: {fr:.3f}")
        if fr > best_face_recog:
            best_face_recog = fr
            save_model = True
            # torch.save(G.state_dict(),
            #            path_dict['GEN_DIR'])
            # torch.save(C.state_dict(), path_dict['DISC_DIR'])
            # print(
            #     f"~~~~Model saved at face recognition: {best_face_recog:.3f}~~~~")

        fid_score = calculate_fid_given_paths(
            (path_dict['OUTPUT_DIR'], VALID_PATH), 50, device, 2048, num_workers=1)
        print(f'fid_score: {fid_score}')
        if fid_score < best_fid_score:
            best_fid_score = fid_score
            save_model = True

            # print(f"Best fid score: {best_fid_score}")
        if save_model:
            torch.save({'Generator': G.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'Discriminator': D.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'epoch': epoch,
                        'best_fid_score': best_fid_score,
                        'best_face_recog': best_face_recog,

                        }, path_dict['MODEL_CHECKPOINT'])
            print(
                f"~~~~~~~Model saved~~~~~~~\n")


if __name__ == "__main__":
    main()
    