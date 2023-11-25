import torch
import torch.nn as nn
import torch.nn.functional as F
from hw3_2_dataset import get_loader, img_transform, CaptionDataset
from hw3_2_model import Pretrained_ViT_Encoder, TransformerDecoder
from torch.nn.utils.rnn import pack_padded_sequence
from tokenizers import Tokenizer
from tqdm import tqdm
import json
import os
from hw3_2_evaluate import evaluate

pretrained_vit_name = "vit_large_patch14_224_clip_laion2b"
hidden_dim = 1024 # should match the encoders hidden dim # 768
load_model = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
train_image_dir = "hw3_data/p2_data/images/train"
val_image_dir = "hw3_data/p2_data/images/val"
train_json_dir = "hw3_data/p2_data/train.json"
val_json_dir = "hw3_data/p2_data/val.json"
tokenizer_dir = "hw3_data/caption_tokenizer.json"
output_json_dir = "output.json"
models_file = "models_file"
model_dir = pretrained_vit_name + "_6layers.pth"

TOKENIZER = Tokenizer.from_file(tokenizer_dir)

num_layers = 6
nhead = 8
dim_feedforward = 4 * hidden_dim # 2048
max_position_embedding = 128
dropout = 0.1
vocab_size = TOKENIZER.get_vocab_size()
START_ID = TOKENIZER.token_to_id("[BOS]")
END_ID = TOKENIZER.token_to_id("[EOS]")
PAD_ID = TOKENIZER.padding["pad_id"]

max_decode_len = 64
beam_size = 6

num_epoch = 100
batch_size = 32
enc_learning_rate = 1e-5
dec_learning_rate = 3e-5


def train_one_epoch(Encoder, Decoder, train_loader, criterion, dec_optimizer, enc_optimizer, train_encoder):
    Encoder.train()
    Decoder.train()
    pbar = tqdm(train_loader)
    for img, cap in pbar:
        img, cap = img.to(device), cap.to(device)

        enc_out = Encoder(img)
        scores = Decoder(cap, enc_out)
        targets = cap[:, 1:]
        target_len = ((cap==END_ID).nonzero()[:,1]).cpu()

        scores = pack_padded_sequence(scores, target_len, batch_first=True, enforce_sorted=False).data.to(device)
        targets = pack_padded_sequence(targets, target_len, batch_first=True, enforce_sorted=False).data.to(device)

        loss = criterion(scores, targets)
        if train_encoder:
            dec_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(Decoder.parameters(), 0.1)
            # nn.utils.clip_grad_norm_(Encoder.parameters(), 0.1)
            dec_optimizer.step()
            enc_optimizer.step()
        else:            
            dec_optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(Decoder.parameters(), 0.1)
            dec_optimizer.step()

        pbar.set_postfix({"loss": loss.item()})

        

def predict(beam_size, max_decode_len, encoder, decoder, val_dataset, output_json_file, device):
    """  val dataset must have batch_size=1  """

    json_dict = {}
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for image, image_name, allcaps in tqdm(val_dataset):
            image = image.unsqueeze(0).to(device)
            cur_beam_size = beam_size
            # Encode
            enc_output = encoder(image)  # (1, 50, encoder_dim)
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
                
                dec_output = decoder(gen_seq, enc_output).to(device) # s x seq_len x vocab_size
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
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != END_ID]
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

            # finish decoding, push rest of the unfinished seqs
            # if len(incomplete_inds) > 0:
            #     complete_seqs.extend(gen_seq[incomplete_inds].tolist())
            #     complete_seqs_scores.extend(scores[incomplete_inds])
            if len(complete_seqs)==0:
                # max_id = scores.index(max(scores))
                max_id = torch.argmax(scores, dim=0)
                predicted_sentence = TOKENIZER.decode(gen_seq[max_id])
            else:
                max_id = complete_seqs_scores.index(max(complete_seqs_scores))
                predicted_sentence = TOKENIZER.decode(complete_seqs[max_id])
            json_dict[image_name] = predicted_sentence
            # print(predicted_sentence)
            # print(allcaps)
        
        json_object = json.dumps(json_dict, indent=4)
        with open(output_json_file, "w") as outfile:
            outfile.write(json_object)

def main():
    # calculate_clip_score(image_dir=val_image_dir, candidates_json="output.json", device=device)
    train_loader = get_loader(image_folder=train_image_dir, json_dir=train_json_dir, img_transform=img_transform, mode="TRAIN", batch_size=batch_size, pad_idx=PAD_ID)
    # val_loader = get_loader(image_folder=val_image_dir, json_dir=val_json_dir, img_transform=img_transform, mode="VAL", batch_size=1, pad_idx=PAD_ID)
    val_dataset = CaptionDataset(data_folder=val_image_dir, json_dir=val_json_dir, transform=img_transform, mode="VAL")

    Encoder = Pretrained_ViT_Encoder(
        model_name=pretrained_vit_name,
        pretrained=True
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

    enc_optimizer = torch.optim.Adam(Encoder.parameters(), lr=enc_learning_rate)
    dec_optimizer = torch.optim.Adam(Decoder.parameters(), lr=dec_learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if load_model:
        print("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(models_file, model_dir), map_location=device)
        Encoder.load_state_dict(checkpoint["Encoder"])
        Decoder.load_state_dict(checkpoint["Decoder"])
        enc_optimizer.load_state_dict(checkpoint["enc_optimizer"])
        dec_optimizer.load_state_dict(checkpoint["dec_optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    
    print("Start training:")
    for epoch in range(start_epoch+1, num_epoch):
        print(f"Epoch: {epoch}")
        train_one_epoch(
            Encoder=Encoder,
            Decoder=Decoder,
            train_loader=train_loader,
            criterion=criterion,
            dec_optimizer=dec_optimizer, 
            enc_optimizer=enc_optimizer, 
            train_encoder=epoch>5
            )
        torch.save({
            "Encoder": Encoder.state_dict(),
            "Decoder": Decoder.state_dict(),
            "enc_optimizer": enc_optimizer.state_dict(),
            "dec_optimizer": dec_optimizer.state_dict(),
            "epoch": epoch,
        }, os.path.join(models_file, model_dir))
        
        if epoch%5 == 0:
            print(f"Start evaluation...")
            predict(
                beam_size=beam_size, 
                max_decode_len=max_decode_len,
                encoder=Encoder,
                decoder=Decoder,
                val_dataset=val_dataset,
                output_json_file=output_json_dir,
                device=device
            )
            evaluate(
                pred_file=output_json_dir,
                images_root=val_image_dir,
                annotation_file=val_json_dir
            )

if __name__ == "__main__":
    main()
        # calculate_clip_score(image_dir=val_image_dir, candidates_json="output.json", device=device)
    # train_loader = get_loader(image_folder=train_image_dir, json_dir=train_json_dir, img_transform=img_transform, mode="TRAIN", batch_size=batch_size, pad_idx=PAD_ID)
    # # val_loader = get_loader(image_folder=val_image_dir, json_dir=val_json_dir, img_transform=img_transform, mode="VAL", batch_size=1, pad_idx=PAD_ID)
    # val_dataset = CaptionDataset(data_folder=val_image_dir, json_dir=val_json_dir, transform=img_transform, mode="VAL")

    # Encoder = Pretrained_ViT_Encoder().to(device)
    # Decoder = TransformerDecoder(
    #     num_layers=num_layers,
    #     vocab_size=vocab_size,
    #     hidden_dim=hidden_dim,
    #     pad_token_id=PAD_ID,
    #     nhead=nhead,
    #     dim_feedforward=dim_feedforward,
    #     max_position_embedding=max_position_embedding,
    #     dropout=dropout,
    #     device=device
    # ).to(device)

    # enc_optimizer = torch.optim.AdamW(Encoder.parameters(), lr=enc_learning_rate)
    # dec_optimizer = torch.optim.AdamW(Decoder.parameters(), lr=dec_learning_rate)
    # criterion = nn.CrossEntropyLoss().to(device)
    
    # if load_model:
    #     print("Loading checkpoint...")
    #     checkpoint = torch.load(model_dir, map_location=device)
    #     Encoder.load_state_dict(checkpoint["Encoder"])
    #     Decoder.load_state_dict(checkpoint["Decoder"])
    #     enc_optimizer.load_state_dict(checkpoint["enc_optimizer"])
    #     dec_optimizer.load_state_dict(checkpoint["dec_optimizer"])
    #     start_epoch = checkpoint["epoch"]
    #     print(f"Start evaluation...")
    #     predict(beam_size=beam_size, 
    #         max_decode_len=max_decode_len,
    #         encoder=Encoder,
    #         decoder=Decoder,
    #         val_dataset=val_dataset,
    #         output_json_file=output_json_dir,
    #         device=device
    #     )
    #     p2_evaluate(pred_file=output_json_dir, images_root=val_image_dir, annotation_file=val_json_dir)
    
