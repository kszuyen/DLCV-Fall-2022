import clip
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

data_dir = ["/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/val/6_490.png", 
            "/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/val/10_490.png", 
            "/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/val/12_490.png"]
json_file_dir = "/home/kszuyen/DLCV/hw3-kszuyen/hw3_data/p1_data/id2label.json"

# Prepare the inputs
with open(json_file_dir) as f:
    label_dict = json.load(f)
# text_inputs = torch.cat([clip.tokenize(f"A photo of a {label_dict[c]}.") for c in label_dict]).to(device)
text_inputs = torch.cat([clip.tokenize(f"A photo of a {label_dict[c]}.") for c in label_dict]).to(device)


for image_dir in data_dir:
    image = Image.open(image_dir)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(5)

    correct_label = label_dict[image_dir.split('/')[-1].split('_')[0]]
    correct_prob = 0
    pred_label_list = []
    prob_list = []
    color_list = []
    # Print the result
    print(f"\nCorrect label: {correct_label}")
    print("Top predictions:\n")
    for i, (value, index) in enumerate(zip(values, indices)):
        print(f"{label_dict[str(index.item())]:>16s}: {100 * value.item():.2f}%")
        pred_label_list.append(label_dict[str(index.item())])
        prob_list.append(100 * value.item())

        if i==0 and label_dict[str(index.item())]!=correct_label:
            color_list.append("red")
        elif label_dict[str(index.item())]==correct_label:
            correct_prob = 100 * value.item()
            color_list.append("green")
        else:
            color_list.append("blue")

    fig = plt.figure()
    fig.add_subplot(121)
    plt.title(f"\nCorrect label: {label_dict[image_dir.split('/')[-1].split('_')[0]]}")
    plt.imshow(image)
    
    fig.add_subplot(122)
    plt.title(f"correct probability: {correct_prob:.2f}%")
    plt.barh(pred_label_list, prob_list, color=color_list)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"clip_output/{correct_label}.png")
    plt.clf()