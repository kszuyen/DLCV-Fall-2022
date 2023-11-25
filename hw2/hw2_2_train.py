import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from hw2_2_utils import cosine_beta_schedule, hw2_2_dataset, linear_beta_schedule, extract, reverse_transform, plot
from hw2_2_models import UNet
from hw2_2_unet import UNET
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import digit_classifier
import os
import matplotlib.pyplot as plt
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_IN_COLAB = True
COLAB_PATH = "/content/drive/MyDrive/hw2-kszuyen"
LOAD_MODEL = True

TRAIN_PATH = 'hw2_data/digits/mnistm/data'
TRAIN_CSV = 'hw2_data/digits/mnistm/train.csv'
path_dict = {
    'OUTPUT': 'diffusion_output',
    'OUTPUT_GRID': 'diffusion_grid',
    'CLASSIFIER_PATH': "Classifier.pth",
    'MODEL_CHECKPOINT': "diffusion.pth"
}

IMAGE_SIZE = 28
IMAGE_CHANNELS = 3
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 2e-4

"""  Define beta schedule  """
TIMESTEPS = 1000
# betas = linear_beta_schedule(timesteps=T)
betas = cosine_beta_schedule(timesteps=TIMESTEPS)

"""  Pre-calculate different terms for closed form  """
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# each value is the previous of alphas_cumprod
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def forward_diffusion_sample(x_start, t, noise=None):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(
        sqrt_alphas_cumprod, t, x_start.shape, device)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape, device
    )

    # return mean + variance
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = forward_diffusion_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


def p_losses(denoise_model, x_start, t, label, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = forward_diffusion_sample(
        x_start=x_start, t=t, noise=noise)

    x_noisy = x_noisy.to(device=device, dtype=torch.float)
    t = t.to(device=device)
    label = label.to(device=device)

    predicted_noise = denoise_model(x_noisy, t, label)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def main():

    if TRAINING_IN_COLAB:
        for path in path_dict:
            path_dict[path] = os.path.join(COLAB_PATH, path_dict[path])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda p: (p * 2) - 1),
    ])
    dataset = hw2_2_dataset(TRAIN_PATH, TRAIN_CSV, transform=transform)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model = UNET(in_ch=IMAGE_CHANNELS, out_ch=IMAGE_CHANNELS)
    model = UNet(TIMESTEPS, num_classes=10, image_size=IMAGE_SIZE, ch=128, ch_mult=[
                 1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    model = model.to(device)
    if LOAD_MODEL:
        print("Loading Checkpoint...")
        checkpoint = torch.load(
            path_dict['MODEL_CHECKPOINT'], map_location=device)
        model.load_state_dict(checkpoint['Diffusion_Model'])
        cur_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f'best acc:{best_acc}')
    else:
        cur_epoch = 0
        best_acc = 0
    # print(model)

    auxiliary_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    C = digit_classifier.Classifier()
    digit_classifier.load_checkpoint(path_dict['CLASSIFIER_PATH'], C)
    C = C.to(device)
    for param in C.parameters():
        param.requires_grad = False

    for epoch in range(cur_epoch+1, NUM_EPOCHS):

        correct = 0
        total = 1e-16
        loop = tqdm(loader, total=len(loader))
        loop.set_description(f'Epoch: {epoch}')
        model.train()
        C.train()
        for step, (image, label) in enumerate(loop):

            optimizer.zero_grad()

            batch_size = image.shape[0]
            image = image.to(device)
            label = label.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, TIMESTEPS, (batch_size,),
                              device=device).long()

            loss = p_losses(model, image, t, label, loss_type="l2")

            """  sampling for aux loss  """
            # start from pure noise (for each example in the batch)
            # img = torch.randn((batch_size, IMAGE_CHANNELS,
            #                    IMAGE_SIZE, IMAGE_SIZE), device=device)
            # t = torch.randint(
            #     0, TIMESTEPS, (batch_size,)).long().to(device)
            img = forward_diffusion_sample(image, t)
            # set_label = torch.randint(
            #     0, 10, (batch_size,)).long().to(device)

            # for i in reversed(range(0, TIMESTEPS)):

            # t = torch.full((batch_size,), i,
            #                device=device, dtype=torch.long)
            # t = torch.randint(
            #     0, TIMESTEPS, (batch_size,)).long().to(device)

            betas_t = extract(betas, t, img.shape, device)
            sqrt_one_minus_alphas_cumprod_t = extract(
                sqrt_one_minus_alphas_cumprod, t, img.shape, device
            )
            sqrt_recip_alphas_t = extract(
                sqrt_recip_alphas, t, img.shape, device)

            # Equation 11 in the paper
            # Use our model (noise predictor) to predict the mean
            model_mean = sqrt_recip_alphas_t * (
                img - betas_t * model(img.float(), t, label) /
                sqrt_one_minus_alphas_cumprod_t
            )

            # posterior_variance_t = extract(
            #     posterior_variance, t, img.shape, device)
            # noise = torch.randn_like(img)
            # # Algorithm 2 line 4:
            # img = model_mean + torch.sqrt(posterior_variance_t) * noise

            pred_label = C(model_mean.float())
            aux_loss = auxiliary_loss(pred_label, label)
            a = torch.randint(0, 10, (1,), device=device)
            total_loss = (a/10) * loss + (1-a/10) * aux_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.)
            optimizer.step()
            """  calculate acc  """

            _, predict = torch.max(pred_label, 1)
            correct += (predict == label).detach().sum().item()
            total += len(predict)

            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'aux_loss': f'{aux_loss.item():.4f}',
                'acc': f'{(correct/total):.3f}',
                'correct/total': f'{correct}/{total})',
                'added': f'{(predict == label).detach().sum().item()}'
            })

            # if step % 5599 == 0 and step != 0:
        """  sampling for output  """
        if epoch % 1 == 0:
            print('\n===> start evaluation ...')
            model.eval()
            C.eval()
            with torch.no_grad():
                # first_img_idx = 0
                index = 0
                for label_idx in range(10):
                    # start from pure noise (for each example in the batch)
                    batch_size = 100
                    img = torch.randn(
                        (batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(device, dtype=torch.float)
                    label = torch.full(
                        (batch_size,), label_idx, device=device, dtype=torch.long)
                    # first_img = torch.empty(
                    #     (6, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

                    ahundred = torch.empty(
                        (100, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device)
                    for i in reversed(range(0, TIMESTEPS)):
                        t = torch.full((batch_size,), i,
                                       device=device, dtype=torch.long)

                        betas_t = extract(betas, t, img.shape, device)
                        sqrt_one_minus_alphas_cumprod_t = extract(
                            sqrt_one_minus_alphas_cumprod, t, img.shape, device
                        )
                        sqrt_recip_alphas_t = extract(
                            sqrt_recip_alphas, t, img.shape, device)

                        # Equation 11 in the paper
                        # Use our model (noise predictor) to predict the mean
                        model_mean = sqrt_recip_alphas_t * (
                            img - betas_t *
                            model(img.float(), t, label) /
                            sqrt_one_minus_alphas_cumprod_t
                        )

                        posterior_variance_t = extract(
                            posterior_variance, t, img.shape, device)
                        noise = torch.randn_like(img)
                        # Algorithm 2 line 4:
                        img = model_mean + \
                            torch.sqrt(posterior_variance_t) * noise
                        if label_idx == 0:
                            if i % (TIMESTEPS/5) == 0:
                                if i != TIMESTEPS:
                                    # first_img[first_img_idx] = img[0]
                                    save_image(img[0], os.path.join(
                                        path_dict['OUTPUT_GRID'], f't_{i}.png'))
                                else:
                                    # first_img[first_img_idx] = model_mean[0]
                                    save_image(model_mean[0], os.path.join(
                                        path_dict['OUTPUT_GRID'], f't_{i}.png'))
                            elif i == (TIMESTEPS-10) or i == (TIMESTEPS-20) or i == (TIMESTEPS-30) or i == (TIMESTEPS-40) or i == (TIMESTEPS-50):
                                save_image(img[0], os.path.join(
                                    path_dict['OUTPUT_GRID'], f't_{i}.png'))

                                # first_img_idx += 1
                    for i in range(batch_size):
                        # plt.savefig(model_mean[i].permute(1, 2, 0), os.path.join(
                        #     path_dict['OUTPUT'], str(label_idx)+'_'+'{0:03}'.format(i)+'.png'))
                        save_image(model_mean[i], os.path.join(
                            path_dict['OUTPUT'], str(label_idx)+'_'+'{0:03}'.format(i)+'.png'))

                    for i in range(10):
                        ahundred[index] = model_mean[i]
                        index += 1
                # image_grid = make_grid(first_img, nrow=6, normalize=False)
                # save_image(image_grid, os.path.join(
                #     path_dict['OUTPUT_GRID'], '0_timestep'+'.png'))
                image_grid = make_grid(
                    ahundred, nrow=10, padding=2, normalize=False)
                save_image(image_grid, os.path.join(
                    path_dict['OUTPUT_GRID'], str(epoch)+'_step'+str(step)+'.png'))

                """  calculate acc  with digit classifier  """
                data_loader = torch.utils.data.DataLoader(
                    digit_classifier.DATA(path_dict['OUTPUT']),
                    batch_size=32,
                    num_workers=4,
                    shuffle=False)

                num_correct = 0
                num_total = 0
                with torch.no_grad():
                    for idx, (imgs, labels) in enumerate(data_loader):
                        imgs, labels = imgs.to(device), labels.to(device)
                        output = C(imgs)
                        _, pred = torch.max(output, 1)
                        num_correct += (pred ==
                                        labels).detach().sum().item()
                        num_total += len(pred)
                print(
                    'acc = {} (correct/total = {}/{})'.format(float(num_correct)/num_total, num_correct, num_total))

            """  save_model  """
            if (num_correct/num_total) > best_acc:
                best_acc = (num_correct/num_total)
                torch.save({'Diffusion_Model': model.state_dict(),
                            # 'optimizer_D': optimizer_D.state_dict(),
                            'epoch': epoch,
                            'step': step,
                            # 'best_fid_score': best_fid_score,
                            # 'best_face_recog': best_face_recog,
                            'best_acc': best_acc,
                            'optimizer': optimizer.state_dict(),

                            }, path_dict['MODEL_CHECKPOINT'])
                print(
                    f"~~~~~~~Model saved~~~~~~~\n")


if __name__ == '__main__':
    main()
