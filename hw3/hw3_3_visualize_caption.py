import torch
import torch.nn.functional as F
from torchvision import transforms
from hw3_2_model import Pretrained_ViT_Encoder, TransformerDecoder
from tokenizers import Tokenizer
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import shutil

import time
start = time.time()

#
p3_image_dir = "hw3_data/p3_data/images"
output_dir = "p3_output"
p2_image_dir = "hw3_data/p2_data/images/val"
lowest_highest_images_name = ["6209779666", "000000392315"]
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

tokenizer_dir = "hw3_data/caption_tokenizer.json" # need to change location
pretrained_vit_name = "vit_large_patch14_224_clip_laion2b"
model_dir = "models_file/vit_large_patch14_224_clip_laion2b.pth"

TOKENIZER = Tokenizer.from_file(tokenizer_dir)

hidden_dim = 1024 # should match the encoders hidden dim 768
num_layers = 8
nhead = 8
dim_feedforward = hidden_dim * 4
max_position_embedding = 128
dropout = 0.1
vocab_size = TOKENIZER.get_vocab_size()
START_ID = TOKENIZER.token_to_id("[BOS]")
END_ID = TOKENIZER.token_to_id("[EOS]")
PAD_ID = TOKENIZER.padding["pad_id"]

max_decode_len = 64
beam_size = 6

ATTENTION_WEIGHT_l1 = None
ATTENTION_WEIGHT_l2 = None
def get_attention_weight(layer="l1"):
    def hook(model, input, output):
        global ATTENTION_WEIGHT_l1, ATTENTION_WEIGHT_l2
        if layer == "l1":
            ATTENTION_WEIGHT_l1 = output[1]
        elif layer == "l2":
            ATTENTION_WEIGHT_l2 = output[1]
    return hook

def img_resize(h, w):
    return transforms.Compose([
        transforms.CenterCrop((min(h, w))),
        transforms.Resize((224, 224))
    ])
transform_totensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def visualize_attention_map(original_image, attention_weight, word, fig, rows, columns, position):
    
    # Attention from the output token to the input space.
    grid_size = int(np.sqrt(attention_weight.size(-1)))
    mask = attention_weight[1:].view(grid_size, grid_size)

    mask = mask.detach().cpu().numpy()
    mask = cv2.resize(mask, original_image.size, interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
    mask /= mask.max()
    mask *= 255
    mask = 255 - mask

    heatmap = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

    result = cv2.addWeighted(
        heatmap, 0.7, 
        np.asarray(original_image).astype(np.uint8), 
        0.3, 0
    )

    fig.add_subplot(rows, columns, position)
    plt.imshow(result)
    plt.title(word)
    plt.axis('off')
def plot_attention_word_by_word(original_image, final_attention_weight, image_name, caption):

    split_sentence = caption.split(" ")
    images_num = len(split_sentence) + 1 # plus original image

    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = images_num // 5 if images_num%5==0 else images_num // 5 + 1

    # plot original image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(original_image)
    plt.axis('off')

    for i in range(len(split_sentence)):
        visualize_attention_map(
            original_image=original_image, 
            attention_weight=final_attention_weight[i+1],
            word=split_sentence[i],
            fig=fig,
            rows=rows,
            columns=columns,
            position=i+2
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, image_name+".png"))
    plt.clf()

def predict_and_visualize_caption(Encoder, Decoder, original_image, image_name):
    image = transform_totensor_normalize(original_image)
    image = image.unsqueeze(0).to(device)
    cur_beam_size = beam_size
    # Encode
    enc_output = Encoder(image)  # (1, 50, encoder_dim)
    b, p, h_dim = enc_output.shape
    # We'll treat the problem as having a batch size of k
    enc_output = enc_output.expand(cur_beam_size, p, h_dim)  # (k, num_pixels, encoder_dim)
    
    # Tensor to store top k previous words at each step; now they're just <start>
    gen_seq = torch.LongTensor([[START_ID]] * cur_beam_size).to(device)  # (k, 1)
    scores = torch.zeros(cur_beam_size, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    attention_weight_list = list()
    complete_attention_weight_list = list()

    # Start decoding
    # b_size is a number less than or equal to beam_size, because sequences are removed from this process once they hit <end>
    for step in range(max_decode_len):
        
        dec_output = Decoder(gen_seq, enc_output).to(device) # s x seq_len x vocab_size
        # average_weight = (ATTENTION_WEIGHT_l1 + ATTENTION_WEIGHT_l2) / 2
        average_weight = ATTENTION_WEIGHT_l2

        prob = F.log_softmax(dec_output[:, -1, :], dim=1) + scores

        # Get the best k candidates from k^2 candidates.
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 0:
            scores, top_k_words = prob[0].topk(cur_beam_size)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            scores, top_k_words = prob.view(-1).topk(cur_beam_size)  # (s)
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        gen_seq = torch.cat([gen_seq[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        if step==0:
            # print(average_weight[prev_word_inds, -1, :].unsqueeze(1).shape)
            attention_weight_list = average_weight[prev_word_inds, -1, :].unsqueeze(1)
        else:
            attention_weight_list = torch.cat(
                [attention_weight_list[prev_word_inds], 
                average_weight[prev_word_inds, -1, :].unsqueeze(1)], dim=1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != END_ID]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(gen_seq[complete_inds].tolist())
            complete_seqs_scores.extend(scores[complete_inds])
            complete_attention_weight_list.extend(attention_weight_list[complete_inds])

        cur_beam_size -= len(complete_inds)  # reduce beam length accordingly
        # Proceed with incomplete sequences
        if cur_beam_size == 0:
            break
        gen_seq = gen_seq[incomplete_inds]
        enc_output = enc_output[prev_word_inds[incomplete_inds]]
        scores = scores[incomplete_inds].unsqueeze(1)
        attention_weight_list = attention_weight_list[incomplete_inds]

    if len(complete_seqs)==0:
        predicted_sentence = ""
    else:
        max_id = complete_seqs_scores.index(max(complete_seqs_scores))
        predicted_sentence = TOKENIZER.decode(complete_seqs[max_id])
        final_attention_weight = complete_attention_weight_list[max_id]
    print(predicted_sentence)

    plot_attention_word_by_word(
        original_image=original_image,
        final_attention_weight=final_attention_weight,
        image_name=image_name,
        caption=predicted_sentence
    )
class dataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.all_images = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)
    def __getitem__(self, index):
        image_name = self.all_images[index]

        image = Image.open(os.path.join(self.image_folder, image_name)).convert("RGB")
        h, w = image.size
        if self.transform:
            image = self.transform(h, w)(image)
        return image, image_name.split(".")[0]

def main():
    test_dataset = dataset(image_folder=p3_image_dir, transform=img_resize)

    Encoder = Pretrained_ViT_Encoder(
        model_name=pretrained_vit_name,
        pretrained=False
    ).to(device)
    Decoder = TransformerDecoder(
        num_layers=num_layers,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        pad_token_id=PAD_ID,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        max_position_embedding=max_position_embedding,
        dropout=dropout,
        device=device
    ).to(device)
    Decoder.layers[-1].multihead_attn.register_forward_hook(get_attention_weight(layer="l1"))
    Decoder.layers[-2].multihead_attn.register_forward_hook(get_attention_weight(layer="l2"))

    checkpoint = torch.load(model_dir, map_location=device)
    Encoder.load_state_dict(checkpoint["Encoder"])
    Decoder.load_state_dict(checkpoint["Decoder"])

    Encoder.eval()
    Decoder.eval()
    with torch.no_grad():
        for original_image, image_name in tqdm(test_dataset):
            predict_and_visualize_caption(
                Encoder=Encoder,
                Decoder=Decoder,
                original_image=original_image,
                image_name=image_name
            )
        for image_name in lowest_highest_images_name:
            original_image = Image.open(os.path.join(p2_image_dir, image_name+".jpg")).convert("RGB")
            h, w = original_image.size
            original_image = img_resize(h, w)(original_image)
            predict_and_visualize_caption(
                Encoder=Encoder,
                Decoder=Decoder,
                original_image=original_image,
                image_name=image_name
            )
        
            

if __name__ == "__main__":
    main()
