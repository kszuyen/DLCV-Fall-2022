from hw2_3_models import CNNModel
from hw2_3_utils import hw2_3_dataset, img_transform
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

source_name = "mnistm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root_dir = "hw2_data/digits"

for target_name in ["svhn", "usps"]:
    source_data = os.path.join(data_root_dir, source_name, "data")
    source_csv = os.path.join(data_root_dir, source_name, "val.csv")
    target_data = os.path.join(data_root_dir, target_name, "data")
    target_csv = os.path.join(data_root_dir, target_name, "val.csv")

    source_img_channels = 3
    target_img_channels = 3 if target_name=="svhn" else 1
    model_dir = f"models_file/dann_mnistm_2_{target_name}.pth"
        
    batch_size = 128
    image_size = 28
    alpha = 0

    model = CNNModel()
    model.to(device)
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint['dann_model'])
    model.eval()

    label_features = None
    domain_features = None
    # forward hook function


    def label_hook(model, input, output):
        global label_features
        label_features = output
        return None
    def domain_hook(model, input, output):
        global domain_features
        domain_features = output
        return None


    # model.feature.f_relu2.register_forward_hook(hook=label_hook)
    model.class_classifier.c_fc2.register_forward_hook(hook=label_hook)
    model.domain_classifier.d_fc1.register_forward_hook(hook=domain_hook)

    # placeholders
    L_FEATS = []
    D_FEATS = []
    PREDS = []
    DOMAIN = []

    for i, (root_dir, csv_file, target_img_channels) in enumerate(zip([source_data, target_data], [source_csv, target_csv], [source_img_channels, target_img_channels])):

        dataset = hw2_3_dataset(root_dir=root_dir, csv_file=csv_file, transform=img_transform(target_img_channels))
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        loop = tqdm(loader, total=len(loader))

        for batch_img, label in loop:
            batch_size = len(batch_img)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            # class_label = torch.LongTensor(batch_size)

            batch_img = batch_img.to(device)
            # t_label = t_label.to(device)
            input_img = input_img.to(device)
            # class_label = class_label.to(device)

            input_img.resize_as_(batch_img).copy_(batch_img)
            # class_label.resize_as_(t_label).copy_(t_label)

            class_output, domain_output = model(input_data=input_img, alpha=alpha)
            preds = class_output.data.max(1, keepdim=True)[1]
            domain = domain_output.data.max(1, keepdim=True)[1]

            L_FEATS.append(label_features.detach().view(-1, 100).cpu().numpy())
            D_FEATS.append(domain_features.detach().view(-1, 100).cpu().numpy())
            PREDS.append(preds.detach().view(-1).cpu().numpy())
            DOMAIN.append(domain.detach().view(-1).cpu().numpy())

    L_FEATS = np.concatenate(L_FEATS)
    D_FEATS = np.concatenate(D_FEATS)
    PREDS = np.concatenate(PREDS)
    DOMAIN = np.concatenate(DOMAIN)

    for (feat_name, FEATS, METHOD, classes) in zip(['label', 'domain'],[L_FEATS, D_FEATS], [PREDS, DOMAIN], [10, 2]):
        # Create a two dimensional t-SNE projection of the embeddings
        tsne = TSNE(2, verbose=1, perplexity=30, init='random')
        tsne_proj = tsne.fit_transform(FEATS)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('Set1')
        colors = [cmap(i) for i in np.linspace(0, 1, classes)]

        fig, ax = plt.subplots(figsize=(8, 8))
        for label in range(classes):
            indices = METHOD == label
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(colors[label]).reshape(1, -1), label=label, alpha=0.5)


        ax.legend(fontsize='xx-small', markerscale=2)
        ax.set_title(f'{target_name}_tsne_{feat_name}')
        plt.savefig(f'tsne/{target_name}_tsne_{feat_name}.png')
        print('finished saving')


    



