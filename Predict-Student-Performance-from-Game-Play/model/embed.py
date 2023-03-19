import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_token, out_token):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=in_token, out_channels=out_token,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        num_embed = {'second': 60, 'minute': 60, 'hour': 24, 'weekday': 7, 'month': 13, 'year': 100}

        Embed = nn.Embedding
        self.second_embed = Embed(num_embed['second'], d_model)
        self.minute_embed = Embed(num_embed['minute'], d_model)
        self.hour_embed = Embed(num_embed['hour'], d_model)
        self.weekday_embed = Embed(num_embed['weekday'], d_model)
        self.month_embed = Embed(num_embed['month'], d_model)
        self.year_embed = Embed(num_embed['year'], d_model)

    def forward(self, x):
        x = x.long()

        second_x = self.second_embed(x[:, :, 5])
        minute_x = self.minute_embed(x[:, :, 4])
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        month_x = self.month_embed(x[:, :, 1])
        year_x = self.year_embed(x[:, :, 0])

        return second_x + minute_x + hour_x + weekday_x + month_x + year_x


class DataEmbedding(nn.Module):
    def __init__(self, token_len, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(token_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
