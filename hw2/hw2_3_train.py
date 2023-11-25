import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from hw2_3_utils import hw2_3_dataset, img_transform
from hw2_3_models import CNNModel
from hw2_3_test import test
import numpy as np
from tqdm import tqdm


# source_name = "mnistm"
source_name = "usps"

target_name = source_name
# target_name = "svhn"
# target_name = "usps"

target_img_channel = 1

TRAINING_IN_COLAB = False
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
IMAGE_SIZE = 28
N_EPOCH = 1000

data_root_dir = "hw2_data/digits/"
COLAB_PATH = "/content/drive/MyDrive/hw2-kszuyen"
path_dict = {
    'model_checkpoint': "models_file/dann_usps_only.pth"
}
if TRAINING_IN_COLAB:
    for path in path_dict:
        path_dict[path] = os.path.join(COLAB_PATH, path_dict[path])
source_data = os.path.join(data_root_dir, source_name, "data")
source_csv = os.path.join(data_root_dir, source_name, "train.csv")
target_data = os.path.join(data_root_dir, target_name, "data")
target_csv = os.path.join(data_root_dir, target_name, "train.csv")

if torch.cuda.is_available():
    device = 'cuda'
    cuda = True
    cudnn.benchmark = True
else:
    device = 'cpu'
    cuda = False

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


source_dataset = hw2_3_dataset(
    root_dir=source_data, csv_file=source_csv, transform=img_transform(img_channels=target_img_channel))
source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)

target_dataset = hw2_3_dataset(
    root_dir=target_data, csv_file=target_csv, transform=img_transform(img_channels=target_img_channel))
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

# load model
model = CNNModel().to(device)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL:
    print('Loading model...')
    checkpoint = torch.load(path_dict['model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['dann_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print('Finished')
else:
    best_acc = 0
    start_epoch = 0

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in model.parameters():
    p.requires_grad = True

# training

for epoch in range(start_epoch+1, N_EPOCH):
    len_dataloader = min(len(source_loader), len(target_loader))
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)
    model.train()
    loop = tqdm(range(len_dataloader), total=len_dataloader)
    loop.set_description(f'Epoch: {epoch}')

    for i in loop:
        p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        model.zero_grad()
        BATCH_SIZE = len(s_label)

        input_img = torch.FloatTensor(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        class_label = torch.LongTensor(BATCH_SIZE)
        domain_label = torch.zeros(BATCH_SIZE)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = model(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        BATCH_SIZE = len(t_img)

        input_img = torch.FloatTensor(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        domain_label = torch.ones(BATCH_SIZE)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = model(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        loop.set_postfix({
            'iter': f'{i}',
            'err_s_label': err_s_label.cpu().data.numpy(),
            'err_s_domain': err_s_domain.cpu().data.numpy(),
            'err_t_domain': err_t_domain.cpu().data.numpy()
        })
        # print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f'
        #       % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
        #          err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    model.eval()
    # source_acc = test(
    #     model=model, root_dir=data_root_dir, dataset_name=source_name)
    target_acc = test(
        model=model, root_dir=data_root_dir, dataset_name=target_name, img_channels=target_img_channel)
    if target_acc > best_acc:
        best_acc = target_acc

        torch.save({'dann_model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,

                    }, path_dict['model_checkpoint'])
        print('~~~~~MODEL SAVED~~~~~')


print('done')
