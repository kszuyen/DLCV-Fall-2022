import os
import clip
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import sys
import csv
# import time
# start = time.time()

class hw3_1_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_images = os.listdir(root_dir)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_name = self.all_images[index]
        image = Image.open(os.path.join(self.root_dir, image_name))

        return image, image_name


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

###
# data_dir = "/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/val"
# json_file_dir = "/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/id2label.json"
# output_dir = "./hw3_1_output.csv"
data_dir = sys.argv[1]
json_file_dir = sys.argv[2]
output_dir = sys.argv[3]
###

# Prepare the inputs
dataset = hw3_1_dataset(root_dir=data_dir)
with open(json_file_dir) as f:
    label_dict = json.load(f)
text_inputs = torch.cat([clip.tokenize(f"A photo of a {label_dict[c]}.") for c in label_dict]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# correct_count = 0

with open(output_dir, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['filename', 'label'])

    for image, image_name in dataset:
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        index = torch.argmax(similarity[0]).item()

        writer.writerow([image_name, index])

        # if index==int(image_name.split("_")[0]):
        #     correct_count += 1

# print(f"Accuracy: {correct_count/len(dataset)}")
# finish = time.time()
# print(f"Took {((finish - start) / 60):.2f} min")