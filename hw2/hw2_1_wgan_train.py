import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from hw2_1_models import Discriminator, Generator, Critic, initialize_weights
from hw2_1_utils import hw2_1_dataset, gradient_penalty
from face_recog import face_recog
import random
# import imageio
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = True

# Hyperparameters
# RANDOM_SEED = 3330
RANDOM_SEED = 3536
NUM_EPOCHS = 500
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64  # to match 1024 from the paper (64x16 = 1024)

LEARNING_RATE_D = 2e-4
LEARNING_RATE_G = 2e-4
BETA1 = 0.5
BETA2 = 0.999
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
# WEIGHT_CLIP = 0.01


path_dict = {
    'TRAIN_PATH': "hw2_data/face/train",
    'VALID_PATH': "hw2_data/face/val",
    'OUTPUT_DIR': "WGANGP_results/output",
    'MODEL_CHECKPOINT': 'models_file/wgangp_checkpoint.pth',
    'STEPGRID': "WGANGP_results/grid",
}


def main():
    print("Random Seed: ", RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.CenterCrop(IMAGE_SIZE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = hw2_1_dataset(
        root_dir=path_dict['TRAIN_PATH'], transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(Z_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)
    C = Critic(IMAGE_CHANNELS, FEATURES_DISC).to(device)

    optimizer_G = optim.Adam(
        G.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    optimizer_C = optim.Adam(
        C.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

    if LOAD_MODEL:
        print("Loading Checkpoint...")
        checkpoint = torch.load(
            path_dict['MODEL_CHECKPOINT'], map_location=device)
        G.load_state_dict(checkpoint['Generator'])
        C.load_state_dict(checkpoint['Critic'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C'])
        best_face_recog = checkpoint['best_face_recog']
        best_fid_score = checkpoint['best_fid_score']
        saved_epoch = checkpoint['epoch']
        print(f"Best face recognition: {best_face_recog:.4f}")
        print(f"Best fid score: {best_fid_score:.4f}")
        print("~~~Finished~~~")
    else:
        G.apply(initialize_weights)
        C.apply(initialize_weights)
        best_face_recog = 0.0
        best_fid_score = 1e6
        saved_epoch = 0

    print("Start training...")

    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)
    for epoch in range(saved_epoch+1, NUM_EPOCHS):
        loop = tqdm(loader, total=len(loader))
        loop.set_description(f'Epoch: {epoch}')
        num_correct = 0
        num_samples = 1e-16
        for batch_idx, real_image in enumerate(loop):
            real_image = real_image.to(device)

            """ Training Phase """
            G.train()
            C.train()

            """  Training Critic  """
            for it in range(CRITIC_ITERATIONS):

                noise = torch.randn(
                    (real_image.shape[0], Z_DIM, 1, 1)).to(device)
                fake_image = G(noise)
                C_real = C(real_image).view(-1)
                C_fake = C(fake_image).view(-1)
                gp = gradient_penalty(C, real_image, fake_image, device=device)

                loss_C = (
                    -(torch.mean(C_real) - torch.mean(C_fake)) + LAMBDA_GP * gp
                )
                C.zero_grad()
                loss_C.backward(retain_graph=True)
                optimizer_C.step()
                ###
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
            output = C(fake_image).view(-1)
            loss_G = -torch.mean(output)
            G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # loop.set_postfix(loss_C=loss_C.item(),
            #                  loss_G=loss_G.item())
            loop.set_postfix({
                'batch': f'{batch_idx}/{len(loader)}',
                'G_loss': torch.mean(loss_G).item(),
                'C_loss': torch.mean(loss_C).item(),
                'real_score': torch.mean(C_real).item(),
                'fake_score': torch.mean(C_fake).item(),
                'C_acc': f'{C_acc:.2f}',
            })
        """  evaluation phase  """
        G.eval()
        C.eval()

        # evaluate with face recognition score
        with torch.no_grad():
            # check_D_accuracy(G, C, loader)

            eval_image = G(fixed_noise)
            image_grid = torchvision.utils.make_grid(
                eval_image, padding=2, normalize=True
            )
            save_image(image_grid, os.path.join(
                path_dict['STEPGRID'], str(epoch)+".png"))
            
            noise = torch.randn(1000, Z_DIM).view(-1, Z_DIM, 1, 1).to(device)
            output_image = G(noise)
            for output_image_count in range(1000):
                save_image(output_image[output_image_count], os.path.join(
                    path_dict['OUTPUT_DIR'], str(output_image_count+1)+".png"))

        fr = face_recog(path_dict['OUTPUT_DIR'])
        print(f"Face recognition: {fr:.3f}")
        if fr > best_face_recog:
            best_face_recog = fr

        fid_score = calculate_fid_given_paths(
            (path_dict['OUTPUT_DIR'], path_dict['VALID_PATH']), 64, device, 2048, num_workers=4)
        print(f'fid_score: {fid_score}')
        if fid_score < best_fid_score:
            best_fid_score = fid_score

            # print(f"Best fid score: {best_fid_score}")
            torch.save({'Generator': G.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'Critic': C.state_dict(),
                        'optimizer_C': optimizer_C.state_dict(),
                        'epoch': epoch,
                        'best_fid_score': best_fid_score,
                        'best_face_recog': best_face_recog,

                        }, path_dict['MODEL_CHECKPOINT'])
            print(
                f"~~~~Model saved at fid score: {best_fid_score:.3f}, face recognition: {fr:.3f}~~~~\n")


if __name__ == "__main__":
    main()
