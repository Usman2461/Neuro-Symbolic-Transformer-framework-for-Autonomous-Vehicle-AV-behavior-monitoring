import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=128, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = out.transpose(1, 2)
        out = self.pool(out).squeeze(-1)
        return out