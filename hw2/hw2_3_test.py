import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from hw2_3_utils import hw2_3_dataset, img_transform
from hw2_3_models import CNNModel


def test(model, root_dir, dataset_name, img_channels=3):

    if torch.cuda.is_available():
        cuda = True
        cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    dataset = hw2_3_dataset(
        root_dir=os.path.join(root_dir, dataset_name, "data"),
        csv_file=os.path.join(root_dir, dataset_name, "val.csv"),
        transform=img_transform(img_channels=img_channels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """ training """

    model = model.eval()

    if cuda:
        model = model.cuda()

    len_dataloader = len(loader)
    data_target_iter = iter(loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = model(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Accuracy of the %s dataset: %f' %
          (dataset_name, accu))
    return accu
