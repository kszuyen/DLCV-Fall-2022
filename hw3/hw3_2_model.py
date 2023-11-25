import torch
import torch.nn as nn
import timm
from typing import Optional
from torch import Tensor

class Pretrained_ViT_Encoder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.pretrained_vit = timm.create_model(model_name, pretrained=pretrained)
        self.pretrained_vit.norm.register_forward_hook(self.get_features())
        self.features = None
    def get_features(self):
        def hook(model, input, output):
            self.features = output
        return hook

    def forward(self, x):
        x = self.pretrained_vit(x)
        return self.features

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, vocab_size, hidden_dim, nhead, dim_feedforward, pad_token_id, max_position_embedding, dropout, device):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.embedding = DecoderEmbeddings(vocab_size=vocab_size,
                                            hidden_dim=hidden_dim,
                                            pad_token_id=pad_token_id,
                                            max_position_embeddings=max_position_embedding,
                                            dropout=dropout,
                                            device=device
        )
        self.device = device
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, enc_out):
        
        B, seq_len = tgt.shape
        tgt_mask = self.generate_square_subsequent_mask(sz=seq_len).to(self.device)

        tgt = self.dropout(self.embedding(tgt))

        for layer in self.layers:
            tgt = layer(tgt, enc_out, tgt_mask)

        out = self.fc_out(tgt)

        return out

class DecoderBlock(nn.Module):

    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, enc_out, tgt_mask: Optional[Tensor] = None):

        tgt2 = self.self_attn(  query=tgt, 
                                key=tgt, 
                                value=tgt, 
                                attn_mask=tgt_mask)[0]

        tgt = self.dropout1(self.norm1(tgt + tgt2))

        tgt2 = self.multihead_attn(query=tgt,
                                   key=enc_out,
                                   value=enc_out)[0]
        
        tgt = self.dropout2(self.norm2(tgt + tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.dropout3(self.norm3(tgt + tgt2))

        return tgt

class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, layer_norm_eps=1e-12, dropout=0.1, device="cpu"):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # position_ids = position_ids.expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "hw3_data/p2_data/images/train"
    json_dir = "hw3_data/p2_data/train.json"
    from hw3_2_dataset import get_loader, img_transform
    loader = get_loader(image_folder=image_dir, json_dir=json_dir, img_transform=img_transform)

    Encoder = Pretrained_ViT_Encoder(
        model_name="vit_large_patch14_224_clip_laion2b",
        pretrained=False
    ).to(device)
    # print(Encoder)
    Decoder = TransformerDecoder(
        num_layers=6,
        vocab_size=18021,
        hidden_dim=1024,
        pad_token_id=0,
        nhead=8,
        dim_feedforward=2048,
        max_position_embedding=64,
        dropout=0.1,
        device=device
    ).to(device)

    # dec_emb = DecoderEmbeddings(vocab_size=18021, hidden_dim=256, pad_token_id=0, max_position_embeddings=64)
    # model = Transformer()
    for img, cap in loader:
        img, cap = img.to(device), cap.to(device)
        print("image shape:", img.shape)
        print("caption shape:", cap.shape)
        enc_out = Encoder(img)
        print("encode output shape:", enc_out.shape)
        out = Decoder(cap, enc_out)
        print("decoder output shape:", out.shape)
        break
