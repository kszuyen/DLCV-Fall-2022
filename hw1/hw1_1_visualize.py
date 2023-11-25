from hw1_1_utils import hw1_1_dataset, transform
from hw1_1_models import hw1_1_CNN
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = hw1_1_CNN()
model.load_state_dict(torch.load(
    '/content/drive/MyDrive/hw1-kszuyen/models_file/hw1_1_cnn_10_4516.pt', map_location=device))
model.to(device)
model.eval()

batch_size = 1
# validation dataset
valid_dataset = hw1_1_dataset(
    '/content/drive/MyDrive/hw1-kszuyen/hw1_data/p1_data/val_50/', 'valid', transform=transform(image_size=32)['valid'])
valid_loader = data.DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False)

features = None
# forward hook function


def hook(model, input, output):
    global features
    features = output
    return None


model.conv7.register_forward_hook(hook=hook)

# feature extraction loop
# placeholders
PREDS = []
FEATS = []
LABEL = []

# loop through batches
for data, targets in valid_loader:

    # move to device
    data = data.to(device)

    # forward pass [with feature extraction]
    scores = model(data)
    _, preds = scores.max(1)  # value, index

    # add feats and preds to lists
    PREDS.append(preds.detach().cpu().numpy())
    FEATS.append(features.detach().view(-1, 1024*2*2).cpu().numpy())
    LABEL.append(targets.numpy())

PREDS = np.concatenate(PREDS)
FEATS = np.concatenate(FEATS)
LABEL = np.concatenate(LABEL)

print(PREDS.shape)
print(FEATS.shape)
print(LABEL.shape)

# Create a two dimensional PCA projection of the embeddings
# pca = PCA(2)
# components = pca.fit_transform(FEATS)


# Create a two dimensional t-SNE projection of the embeddings
tsne = TSNE(2, verbose=1, perplexity=45, init='random')
tsne_proj = tsne.fit_transform(FEATS)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, 50)]

fig, ax = plt.subplots(figsize=(8, 8))
for label in range(50):
    indices = PREDS == label
    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1],
               c=np.array(colors[label]).reshape(1, -1), label=label, alpha=0.5)
    # ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
    #     cmap(label)).reshape(1, 4), label=label, alpha=0.5)
ax.legend(fontsize='xx-small', markerscale=2)
ax.set_title("2d_tsne_pred")
plt.savefig(
    '/content/drive/MyDrive/hw1-kszuyen/cnn_visualize_embeddings/2d_tsne_pred.png')
print('finished saving: 2d_tsne_pred.png')

fig, ax = plt.subplots(figsize=(8, 8))
for label in range(50):
    indices = LABEL == label
    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1],
               c=np.array(colors[label]).reshape(1, -1), label=label, alpha=0.5)
    # ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
    #     cmap(label)).reshape(1, 4), label=label, alpha=0.5)
ax.legend(fontsize='xx-small', markerscale=2)
ax.set_title("2d_tsne_label")
plt.savefig(
    '/content/drive/MyDrive/hw1-kszuyen/cnn_visualize_embeddings/2d_tsne_label.png')
print('finished saving: 2d_tsne_label.png')

# # Create 3 dimensional tsne
# tsne = TSNE(3, verbose=1, perplexity=45, init='random')
# tsne_proj = tsne.fit_transform(FEATS)

# fig, ax = plt.subplots(figsize=(8, 8))
# for label in range(50):
#     indices = PREDS == label
#     ax.scatter(tsne_proj[indices, 0],
#                tsne_proj[indices, 1],
#                tsne_proj[indices, 2],
#                color=np.array(colors[label]).reshape(1, -1),
#                label=label,
#                alpha=0.5)
# ax.set_title("3d_tsne_pred")
# ax.legend(fontsize='xx-small', markerscale=2)
# plt.savefig('3d_tsne_pred.pdf', format='pdf')

# fig, ax = plt.subplots(figsize=(8, 8))
# for label in range(50):
#     indices = LABEL == label
#     ax.scatter(tsne_proj[indices, 0],
#                tsne_proj[indices, 1],
#                tsne_proj[indices, 2],
#                color=np.array(colors[label]).reshape(1, -1),
#                label=label,
#                alpha=0.5)
# ax.set_title("3d_tsne_label")
# ax.legend(fontsize='xx-small', markerscale=2)
# plt.savefig('3d_tsne_label.pdf', format='pdf')

# pca
# Create a two dimensional t-SNE projection of the embeddings
pca = PCA(2, svd_solver='randomized')
pca_proj = pca.fit_transform(FEATS)
# Plot those points as a scatter plot and label them based on the pred labels

fig, ax = plt.subplots(figsize=(8, 8))
for label in range(50):
    indices = PREDS == label
    ax.scatter(pca_proj[indices, 0], pca_proj[indices, 1],
               c=np.array(colors[label]).reshape(1, -1), label=label, alpha=0.5)
    # ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
    #     cmap(label)).reshape(1, 4), label=label, alpha=0.5)
ax.legend(fontsize='xx-small', markerscale=2)
ax.set_title("2d_pca_pred")
plt.savefig(
    '/content/drive/MyDrive/hw1-kszuyen/cnn_visualize_embeddings/2d_pca_pred.png')
print('finished saving: 2d_pca_pred.png')

fig, ax = plt.subplots(figsize=(8, 8))
for label in range(50):
    indices = LABEL == label
    ax.scatter(pca_proj[indices, 0], pca_proj[indices, 1],
               c=np.array(colors[label]).reshape(1, -1), label=label, alpha=0.5)
    # ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
    #     cmap(label)).reshape(1, 4), label=label, alpha=0.5)
ax.legend(fontsize='xx-small', markerscale=2)
ax.set_title("2d_pca_label")
plt.savefig(
    '/content/drive/MyDrive/hw1-kszuyen/cnn_visualize_embeddings/2d_pca_label.png')
print('finished saving: 2d_pca_label.png')

# # Create 3 dimensional tsne
# pca = PCA(3)
# pca_proj = pca.fit_transform(FEATS)

# fig, ax = plt.subplots(figsize=(8, 8))
# for label in range(50):
#     indices = PREDS == label
#     ax.scatter(pca_proj[indices, 0],
#                pca_proj[indices, 1],
#                pca_proj[indices, 2],
#                c=np.array(colors[label]).reshape(1, -1),
#                label=label,
#                alpha=0.5)
# ax.set_title("3d_pca_pred")
# ax.legend(fontsize='xx-small', markerscale=2)
# plt.savefig('3d_pca_pred.pdf', format='pdf')

# fig, ax = plt.subplots(figsize=(8, 8))
# for label in range(50):
#     indices = LABEL == label
#     ax.scatter(pca_proj[indices, 0],
#                pca_proj[indices, 1],
#                pca_proj[indices, 2],
#                color=np.array(colors[label]).reshape(1, -1),
#                label=label,
#                alpha=0.5)
# ax.set_title("3d_pca_label")
# ax.legend(fontsize='xx-small', markerscale=2)
# plt.savefig('3d_pca_label.pdf', format='pdf')
