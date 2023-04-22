import torch
import torch.nn as nn
from blocks.attention import FullAttention, ProbAttention, AttentionLayer
from blocks.embed import DataEmbedding
from blocks.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack


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
            nn.Conv1d(in_channels=seq_len // 2**(e_layers-1), out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 3, bias=True))   # [3, 10, 5]
        self.judge2 = nn.Sequential(
            # B * C * L <--> B * L * D
            nn.Conv1d(in_channels=seq_len // 2 ** (e_layers - 1), out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 10, bias=True))
        self.judge3 = nn.Sequential(
            # B * C * L <--> B * L * D
            nn.Conv1d(in_channels=seq_len // 2 ** (e_layers - 1), out_channels=1, kernel_size=1, padding=0),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_model, 5, bias=True))

    def forward(self, x, enc_self_mask=None):
        enc_out = self.enc_embedding(x)
        enc_out1, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out2, _ = self.encoder2(enc_out1, attn_mask=enc_self_mask)
        enc_out3, _ = self.encoder3(enc_out2, attn_mask=enc_self_mask)
        out1 = self.judge1(enc_out1)
        out2 = self.judge2(enc_out2)
        out3 = self.judge3(enc_out3)
        out1, out2, out3 = torch.squeeze(out1), torch.squeeze(out2), torch.squeeze(out3)
        return [out1, out2, out3]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
