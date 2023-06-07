import torch
import torch.nn as nn
from blocks.attention import FullAttention, ProbAttention, AttentionLayer
from blocks.embed import DataEmbedding
from blocks.encoder import Encoder, EncoderLayer, ConvLayer


class HFAttention(nn.Module):
    def __init__(self, seq_len, token_size, d_model=512,
                 factor=5, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', distil=True):
        super(HFAttention, self).__init__()
        self.attn = attn
        # Encoding
        self.enc_embedding = DataEmbedding(token_size, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder2, self.encoder3 = [Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for _ in range(2)]

        self.judge1 = nn.Sequential(
            # B * C * L <--> B * L * D
            nn.Conv1d(in_channels=seq_len // 2**(e_layers-1) +1, out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 3, bias=True))   # [3, 10, 5]
        self.judge2 = nn.Sequential(
            # B * C * L <--> B * L * D
            nn.Conv1d(in_channels=seq_len // 2 ** (e_layers - 1) + 1, out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 10, bias=True))
        self.judge3 = nn.Sequential(
            # B * C * L <--> B * L * D
            nn.Conv1d(in_channels=seq_len // 2 ** (e_layers - 1) + 1, out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 5, bias=True))

        self.stamp_embedding = nn.Sequential(
            nn.Linear(4, 4*d_model),  # 4 denote [year, month, weekday, hour]
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x, stamp, enc_self_mask=None):
        stamp = self.stamp_embedding(stamp)  # (B, 4)-->(B, D) D=d_model
        enc_out = self.enc_embedding(x)  # (B, L, T)-->(B, L, D) T=token_size, D=d_model
        enc_out = torch.cat([enc_out, stamp.unsqueeze(1)], dim=1)
        enc_out1, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out2, _ = self.encoder2(enc_out1, attn_mask=enc_self_mask)
        enc_out3, _ = self.encoder3(enc_out2, attn_mask=enc_self_mask)
        out1 = self.judge1(enc_out1)
        out2 = self.judge2(enc_out2)
        out3 = self.judge3(enc_out3)
        out1, out2, out3 = torch.squeeze(out1), torch.squeeze(out2), torch.squeeze(out3)
        return [out1, out2, out3]

