import torch
import torch.nn as nn
from hw1_2_utils import hw1_2_dataset, augmentation, label2mask, save_pred
from hw1_2_models import vgg16_fcn32, Unet, vgg16_fcn8
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from mean_iou_evaluate import mean_iou_score, read_masks
import shutil

# set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# files
train_file = '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p2_data/train'
valid_file = '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p2_data/validation'
pred_file = '/content/drive/MyDrive/hw1-kszuyen/pred_masks'
hw1_2_outputfile = '/content/drive/MyDrive/hw1-kszuyen/hw1_2_outputfile'
# train_file = 'hw1_data/p2_data/train'
# valid_file = 'hw1_data/p2_data/validation'
# pred_file = 'pred_masks'
# hw1_2_outputfile = 'hw1_2_outputfile'
# Hyperparameters
batch_size = 8
num_epochs = 100
learning_rate = 1e-5
# weights = torch.tensor([2.1500,   1.0000,   3.4753,
#                         1.6845,   4.7583,   1.4436, 155.7826], device=device)

load_model = True
model_location = '/content/drive/MyDrive/hw1-kszuyen/models_file/vgg16_fcn8.pt'


def train(model, loader, optimizer, criterion, num_epochs, pred_file):
    val_meaniou_history = []
    best_meaniou = 0.7337003749015627

    for epoch in range(num_epochs):
        image_count = 0
        for phase in ['train', 'valid']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            loop = tqdm(loader[phase], total=len(loader[phase]))
            for batch_idx, (image, mask) in enumerate(loop):
                image, mask = image.to(device), mask.to(device)
                # mask shape: (512*512)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    """ forward """
                    pred = model(image)
                    mask = mask.squeeze(1).type(torch.LongTensor).to(device)
                    loss = criterion(pred, mask)

                    if phase == 'train':
                        """backward"""
                        loss.backward()

                        """gradient descent"""
                        optimizer.step()

                loop.set_description(f'Epoch: {epoch}')
                loop.set_postfix(loss=loss.item())

                if phase == 'valid':
                    _, predicted = torch.max(pred.data, 1)
                    predicted = predicted.detach().cpu().numpy()
                    for i in range(len(predicted)):
                        pred_mask = label2mask(predicted[i, :, :])
                        image_index = "%04d" % image_count

                        # save mask for homework
                        # if epoch % 4 == 0 and image_index in {'0013', '0062', '0104'}:
                        #     save_pred(hw1_2_outputfile, pred_mask,
                        #               str(epoch)+'_'+image_index)

                        save_pred(pred_file, pred_mask, image_index)
                        image_count += 1

        print(f'Epoch {epoch}:')
        pr = read_masks(pred_file)
        gt = read_masks(valid_file)
        epoch_meaniou = mean_iou_score(pred=pr, labels=gt)
        val_meaniou_history.append(epoch_meaniou)

        if epoch_meaniou > best_meaniou:
            best_meaniou = epoch_meaniou
            # torch.save(model.state_dict(
            # ), '/content/drive/MyDrive/hw1-kszuyen/models_file/vgg16_fcn32.pt')
            torch.save(model.state_dict(),
                       model_location)
            print(f'Model saved at val meaniou = {best_meaniou}')

    print(f'Best meaniou: {best_meaniou:4f}')
    return val_meaniou_history


def main():
    """load model"""
    # model = vgg16_fcn32()
    # model = Unet()
    model = vgg16_fcn8()
    if load_model:
        model.load_state_dict(torch.load(model_location, map_location=device))
        print('Model loaded from: ', model_location)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate)

    # load dataset
    train_dataset = hw1_2_dataset(train_file, transform=augmentation)
    valid_dataset = hw1_2_dataset(valid_file, transform=None)
    # print(train_dataset.__len__())
    # print(valid_dataset.__len__())
    # loader
    dataloader = {
        'train':
        DataLoader(dataset=train_dataset,
                   batch_size=batch_size, shuffle=True),
        'valid':
        DataLoader(dataset=valid_dataset,
                   batch_size=batch_size, shuffle=False)
    }

    if os.path.exists(pred_file):
        shutil.rmtree(pred_file)
    os.makedirs(pred_file)
    if os.path.exists(hw1_2_outputfile):
        shutil.rmtree(hw1_2_outputfile)
    os.makedirs(hw1_2_outputfile)
    train(model, dataloader, optimizer=optimizer, criterion=criterion,
          num_epochs=num_epochs, pred_file=pred_file)


if __name__ == '__main__':
    main()
