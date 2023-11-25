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
import sys

# from hw3_2_evaluate import evaluate
# import time
# start = time.time()

#
# val_image_dir = "hw3_data/p2_data/images/val"
# output_json_dir = "output.json"
# val_json_dir = "hw3_data/p2_data/val.json"
#

val_image_dir = sys.argv[1]
output_json_dir = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_dir = "caption_tokenizer.json" # need to change location

# model_dir = "models_file/vit_p32.pth"
# pretrained_vit_name = "vit_base_patch32_224_in21k"
pretrained_vit_name = "vit_large_patch14_224_clip_laion2b"
# model_dir = "models_file/vit_large_patch14_224_clip_laion2b.pth"
model_dir = "vit_large_patch14_224_clip_laion2b.pth"

TOKENIZER = Tokenizer.from_file(tokenizer_dir)

# hidden_dim = 768 # should match the encoders hidden dim
hidden_dim = 1024
num_layers = 8
nhead = 8
# dim_feedforward = 2048
dim_feedforward = hidden_dim * 4
max_position_embedding = 128
dropout = 0.1
vocab_size = TOKENIZER.get_vocab_size()
START_ID = TOKENIZER.token_to_id("[BOS]")
END_ID = TOKENIZER.token_to_id("[EOS]")
PAD_ID = TOKENIZER.padding["pad_id"]

max_decode_len = 64
beam_size = 6

def img_transform(h, w):
    return transforms.Compose([
        transforms.CenterCrop((min(h, w))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

# val_dataset = CaptionDataset(data_folder=val_image_dir, json_dir=val_json_dir, transform=img_transform, mode="VAL")
test_dataset = dataset(image_folder=val_image_dir, transform=img_transform)

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

checkpoint = torch.load(model_dir, map_location=device)
Encoder.load_state_dict(checkpoint["Encoder"])
Decoder.load_state_dict(checkpoint["Decoder"])

json_dict = {}
Encoder.eval()
Decoder.eval()
with torch.no_grad():
    for image, image_name in tqdm(test_dataset):
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

        # Start decoding
        # b_size is a number less than or equal to beam_size, because sequences are removed from this process once they hit <end>
        for step in range(max_decode_len):
            
            dec_output = Decoder(gen_seq, enc_output).to(device) # s x seq_len x vocab_size
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
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != END_ID]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(gen_seq[complete_inds].tolist())
                complete_seqs_scores.extend(scores[complete_inds])
            cur_beam_size -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if cur_beam_size == 0:
                break
            gen_seq = gen_seq[incomplete_inds]
            enc_output = enc_output[prev_word_inds[incomplete_inds]]
            scores = scores[incomplete_inds].unsqueeze(1)

        # finish decoding
        if len(complete_seqs)==0:
            # max_id = scores.index(max(scores.tolist()))
            max_id = torch.argmax(scores, dim=0)
            predicted_sentence = TOKENIZER.decode(gen_seq.tolist()[max_id])
        else:
            max_id = complete_seqs_scores.index(max(complete_seqs_scores))
            predicted_sentence = TOKENIZER.decode(complete_seqs[max_id])
        json_dict[image_name] = predicted_sentence
        # print(predicted_sentence)

    json_object = json.dumps(json_dict, indent=4)
    with open(output_json_dir, "w") as outfile:
        outfile.write(json_object)

# finish = time.time()
# print("took ", (finish-start)/60, "min." )

# evaluate(
#     pred_file=output_json_dir,
#     images_root=val_image_dir,
#     annotation_file=val_json_dir
# )
