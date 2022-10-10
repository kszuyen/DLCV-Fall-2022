import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from hw1_1_utils import hw1_1_dataset, transform
from hw1_1_models import hw1_1_CNN, hw1_1_densenet
from tqdm import tqdm

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 64
num_epochs = 200
learning_rate = 1e-4
# weight_decay = 0.00001  # L2 normalization
# momentum = 0.5


train_dataset = hw1_1_dataset('/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/train_50', 'train',
                              transform=transform(image_size=32)['train'])
train_dataloader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = hw1_1_dataset(
    '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/val_50/', 'val', transform=transform(image_size=32)['valid'])
test_dataloader = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_withoutda_dataset = hw1_1_dataset(
    '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/train_50/', 'train', transform=transform(image_size=32)['valid'])
train_withoutda_loader = data.DataLoader(
    dataset=train_withoutda_dataset, batch_size=batch_size, shuffle=False)

# Model
model = hw1_1_CNN()
# print(model)
# model = hw1_1_densenet()
model.to(device)
# # load model
# model.load_state_dict(torch.load(
#     'best_checkpoint_hw1_1_cnn.pt', map_location=device))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)


def check_accuracy(dataloader, model):
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)  # value, index

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'Got {num_correct}/{num_samples}, with accuracy {float(num_correct)/float(num_samples):.2f}')

    return num_correct / num_samples


# Training session
best_acc = 0
best_epoch = 0
for epoch in range(num_epochs):
    model.train()
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    loop = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_idx, (data, targets) in enumerate(loop):
        # get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

        loop.set_description(f'Epoch: {epoch}')
        loop.set_postfix(loss=loss.item())
    # if (epoch+1) % 5 == 0:
    # check_accuracy(train_dataloader, model)
    # check_accuracy(train_withoutda_loader, model)
    test_acc = check_accuracy(test_dataloader, model)
    if test_acc > best_acc:
        torch.save(model.state_dict(
        ), '/content/drive/MyDrive/hw1-kszuyen/models_file/hw1_1_cnn.pt')
        print(f'Model saved at acc= {test_acc}')
        best_acc = test_acc
        best_epoch = epoch
    # for hw
    if epoch % 10 == 0:
        torch.save(model.state_dict(
        ), '/content/drive/MyDrive/hw1-kszuyen/models_file/hw1_1_cnn_'+str(epoch)+'_'+str(test_acc)+'.pt')

model.eval()
check_accuracy(train_dataloader, model)
check_accuracy(train_withoutda_loader, model)
check_accuracy(test_dataloader, model)
print(f'Best epoch: {best_epoch}')
