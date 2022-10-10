import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, transforms
from hw1_1_utils import hw1_1_dataset, hw1_1_test_dataset, transform, hw1_1_densenet
from tqdm import tqdm

# file path
train_file = '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/train_50'
valid_file = '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/val_50/'
model_path = '/content/drive/MyDrive/hw1-kszuyen/models_file/hw1_1_densenet.pt'
# train_file = 'hw1_data/p1_data/train_50'
# valid_file = 'hw1_data/p1_data/val_50/'
# model_path = 'models_file/hw1_1_densenet.pt'

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_model = False
# hyperparameters
batch_size = 64
num_epochs = 200
learning_rate = 1e-4
# weight_decay = 1e-6  # apply L2-norm


def train_model(model, loader, criterion, optimizer, num_epochs=200):
    # since = time.time()

    val_acc_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            num_corrects = 0
            num_samples = 0

            loop = tqdm(loader[phase], total=len(loader[phase]))
            # Iterate over data.
            for data, targets in loop:
                data = data.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    scores = model(data)
                    loss = criterion(scores, targets)

                    _, preds = scores.max(1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loop.set_description(f'Epoch: {epoch}')
                loop.set_postfix(loss=loss.item())
                # loop.set_postfix(acc=epoch_acc)
                # statistics
                running_loss += loss.item() * data.size(0)
                # num_corrects += torch.sum(preds == targets.data)
                num_corrects += (preds == targets).sum()
                num_samples += preds.size(0)

            epoch_loss = running_loss / len(loader[phase].dataset)
            epoch_acc = num_corrects / num_samples

            # print and save accuracy
            if phase == 'valid':
                print(f'Accuracy: {epoch_acc}')
                val_acc_history.append(epoch_acc)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                print(f'Accuracy: {epoch_acc}')
                best_acc = epoch_acc
                torch.save(model, model_path)
                # torch.save(model.state_dict(),
                #            model_path)
                print(f'Model saved at val acc = {epoch_acc}')

        print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(torch.load(
    # ))
    return val_acc_history


def main():

    # model = hw1_1_densenet()
    # if load_model:
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 1000),
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1000, 500),
        nn.BatchNorm1d(500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 50)
    )
    model.to(device)

    train_dataset = hw1_1_dataset(train_file, 'train',
                                  transform=transform(image_size=224)['train'])
    valid_dataset = hw1_1_dataset(
        valid_file, 'valid',
        transform=transform(image_size=224)['valid'])

    loader = {
        'train':
            data.DataLoader(dataset=train_dataset,
                            batch_size=batch_size, shuffle=True),
        'valid':
            data.DataLoader(dataset=valid_dataset,
                            batch_size=batch_size, shuffle=False)
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_model(model, loader,
                criterion, optimizer, num_epochs)


if __name__ == '__main__':
    main()
