import torch
import torch.nn as nn

class SymbolicBridge(nn.Module):
    def __init__(self, embed_dim=256, num_predicates=10):
        super(SymbolicBridge, self).__init__()
        self.linear = nn.Linear(embed_dim, num_predicates)
        self.activation = nn.Sigmoid()

    def forward(self, embeddings):
        preds = self.linear(embeddings)
        return self.activation(preds)