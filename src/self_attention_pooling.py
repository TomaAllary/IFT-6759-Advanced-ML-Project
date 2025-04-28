import math
import torch
import torch.nn as nn

class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dim)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.norm(x)
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Softmax over seq_len
        weighted = attn_weights * x  # (batch_size, seq_len, input_dim)
        pooled = torch.sum(weighted, dim=1)  # (batch_size, input_dim)
        return pooled
