import torch.nn.functional as F
from torch import Tensor, nn
import torch
from config import *
from typing import Any, List


class TConvLayer(nn.Module):
    """Dilated temporal convolution layer.

    Considering the time cost, I currently disable dilation.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            kernel_size: int,
            dilation: int,
            bias: bool = True,
            act: str = "relu",
            dropout: float = 0.1,
    ):
        super(TConvLayer, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        # Model blocks
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation, bias=bias),
            dim=None
        )
        self.bn = nn.BatchNorm1d(out_dim)
        if act == "relu":
            self.act = nn.ReLU()
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, C', P)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class EventAwareEncoder(nn.Module):
    """Event-aware encoder based on 1D-Conv."""

    def __init__(
            self,
            h_dim: int = 128,
            out_dim: int = 128,
            readout: bool = True,
            cat_feats: List[str] = ["event_comb_code", "room_fqid_code"]
    ):
        super(EventAwareEncoder, self).__init__()

        # Network parameters
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.cat_feats = cat_feats

        # Model blocks
        # Categorical embeddings
        self.embs = nn.ModuleList()
        for cat_feat in cat_feats:
            self.embs.append(nn.Embedding(CAT_FEAT_SIZE[cat_feat] + 1, 32, padding_idx=0))
        self.emb_enc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        # Feature extractor
        self.convs = nn.ModuleList()
        for l, (dilation, kernel_size) in enumerate(zip([2 ** i for i in range(3)], [7, 7, 5])):
            self.convs.append(TConvLayer(64, h_dim, kernel_size, dilation=1))  # No dilation
        # Readout layer
        if readout:
            self.readout = nn.Sequential(
                nn.Linear(2 * (h_dim // 2), out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
        else:
            self.readout = None

    def forward(self, x: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, P, C)
            x_cat: (B, P, M)
        """

        # Categorical embeddings
        x_cat = x_cat + 1
        x_emb = []
        for i in range(len(self.cat_feats)):
            x_emb.append(self.embs[i](x_cat[..., i]))  # (B, P, emb_dim)
        x_emb = torch.cat(x_emb, dim=-1)  # (B, P, C')
        x_emb = self.emb_enc(x_emb) + x_emb  # (B, P, C')
        x = x * x_emb  # (B, P, C')
        x = self.dropout(x)

        # Feature extractor
        x = x.transpose(1, 2)  # (B, C', P)
        x_skip = []
        for l in range(3):
            x_conv = self.convs[l](x)  # (B, C' * 2, P')
            x_filter, x_gate = torch.split(x_conv, x_conv.size(1) // 2, dim=1)
            x_conv = torch.tanh(x_filter) * torch.sigmoid(x_gate)  # (B, C', P')

            x_conv = self.dropout(x_conv)

            # Skip connection
            x_skip.append(x_conv.unsqueeze(dim=1))  # (B, L (1), C', P')

            x = x_conv

        # Process skipped latent representation
        for l in range(3 - 1):
            x_skip[l] = x_skip[l][..., -x_skip[-1].size(3):]
        x_skip = torch.cat(x_skip, dim=1)  # (B, L, C', P_truc)
        x_skip = torch.sum(x_skip, dim=1)  # (B, C', P_truc)

        # Readout layer
        if self.readout is not None:
            x_std = torch.std(x_skip, dim=-1)  # Std pooling
            x_mean = torch.mean(x_skip, dim=-1)  # Mean pooling
            x = torch.cat([x_std, x_mean], dim=1)
            x = self.readout(x)  # (B, out_dim)

        return x


class StampEmbed(nn.Module):
    def __init__(self, num_embeddings: list, embedding_dim: int):
        super(StampEmbed, self).__init__()
        self.year_embed = nn.Embedding(num_embeddings[0], embedding_dim)
        self.month_embed = nn.Embedding(num_embeddings[1], embedding_dim)
        self.weekday_embed = nn.Embedding(num_embeddings[2], embedding_dim)
        self.hour_embed = nn.Embedding(num_embeddings[3], embedding_dim)

    def forward(self, year, month, weekday, hour):
        x = self.year_embed(year) + self.month_embed(month) + self.weekday_embed(weekday) + self.hour_embed(hour)
        return x


class EventConvSimple(nn.Module):

    def __init__(self, n_lvs: int, out_dim: int, **model_cfg: Any):
        self.name = self.__class__.__name__
        super(EventConvSimple, self).__init__()

        enc_out_dim = 128

        # Network parameters
        self.n_lvs = n_lvs
        self.out_dim = out_dim
        self.cat_feats = model_cfg["cat_feats"]

        self.encoder = EventAwareEncoder(h_dim=128, out_dim=enc_out_dim, cat_feats=self.cat_feats)
        num_embeddings = [23, 12, 7, 24]
        self.stamp_embed = StampEmbed(num_embeddings, enc_out_dim)
        self.clf = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_out_dim // 2, out_dim),
        )

    def forward(self, x: Tensor, x_cat: Tensor, stamp) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, P, C)
            x_cat: (B, P, M)
        """
        x = self.encoder(x, x_cat)
        stamp = self.stamp_embed(stamp[:, 0], stamp[:, 1], stamp[:, 2], stamp[:, 3])
        x = torch.add(x, stamp)
        x = self.clf(x)

        return x
