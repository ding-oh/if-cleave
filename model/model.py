import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence modeling"""

    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: [batch, seq_len, dim]
        seq_len = x.shape[1]
        dim = x.shape[2]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_emb = emb.cos()[None, :, :dim]
        sin_emb = emb.sin()[None, :, :dim]

        # Apply rotary embeddings
        x1 = x[..., :dim//2]
        x2 = x[..., dim//2:]

        x_rope = torch.cat([
            x1 * cos_emb[..., :dim//2] - x2 * sin_emb[..., dim//2:],
            x1 * sin_emb[..., :dim//2] + x2 * cos_emb[..., dim//2:]
        ], dim=-1)

        return x_rope


class GatedFusion(nn.Module):
    """Gated mechanism to fuse features"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_input, x_hidden):
        # x_input: original features, x_hidden: learned features
        gate_input = torch.cat([x_input, x_hidden], dim=-1)
        gate_value = self.gate(gate_input)

        transformed_input = self.transform(x_input)
        fused = gate_value * x_hidden + (1 - gate_value) * transformed_input

        return fused


class BiLSTMGNN(nn.Module):
    """Advanced BiLSTM with RoPE, Attention, and Gated Fusion"""

    def __init__(self, input_dim=518, hidden_dim=256, output_dim=1,
                 num_layers=2, num_heads=8, dropout=0.2, use_batchnorm=False):
        super(BiLSTMGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # RoPE for positional encoding
        self.rope = RotaryPositionEmbedding(hidden_dim)

        # Pre-LSTM attention
        self.pre_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pre_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # BiLSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.post_lstm_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Gated fusion for original features
        self.gated_fusion = GatedFusion(input_dim, hidden_dim)

        # Final layers
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.long_skip_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch=None):
        # x: [N, 518]
        x_orig = x  # Preserve original features

        # Project input
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        x_proj = x

        if batch is not None:
            batch_size = batch.max().item() + 1
            outputs = []

            for b in range(batch_size):
                mask = (batch == b)
                x_batch = x_proj[mask].unsqueeze(0)  # [1, seq_len, hidden_dim]
                x_proj_batch = x_proj[mask]  # [seq_len, hidden_dim]
                x_orig_batch = x_orig[mask]

                # Apply RoPE
                x_batch = self.rope(x_batch)

                # Pre-LSTM attention with residual
                attn_out, _ = self.pre_attention(x_batch, x_batch, x_batch)
                x_batch = self.pre_attn_norm(x_batch + self.dropout(attn_out))

                # BiLSTM with residual
                residual = x_batch
                lstm_out, _ = self.lstm(x_batch)
                x_batch = self.post_lstm_norm(lstm_out + residual)

                # Gated fusion with original features
                x_batch = x_batch.squeeze(0)  # [seq_len, hidden_dim]
                x_batch = self.gated_fusion(x_orig_batch, x_batch)

                # FFN with residual
                residual = x_batch
                ffn_out = self.ffn(x_batch)
                x_batch = self.ffn_norm(residual + self.dropout(ffn_out))
                x_batch = self.long_skip_norm(x_batch + x_proj_batch)

                outputs.append(x_batch)

            x = torch.cat(outputs, dim=0)
        else:
            # Single sequence (inference mode)
            x = x_proj.unsqueeze(0)  # [1, N, hidden_dim]

            x = self.rope(x)

            attn_out, _ = self.pre_attention(x, x, x)
            x = self.pre_attn_norm(x + self.dropout(attn_out))

            residual = x
            lstm_out, _ = self.lstm(x)
            x = self.post_lstm_norm(lstm_out + residual)

            x = x.squeeze(0)
            x = self.gated_fusion(x_orig, x)

            residual = x
            ffn_out = self.ffn(x)
            x = self.ffn_norm(residual + self.dropout(ffn_out))
            x = self.long_skip_norm(x + x_proj)

        # Output projection
        x = self.output_proj(x)
        return x.squeeze(-1)


def get_model(model_name, **kwargs):
    """Factory function to get model by name"""
    models = {
        'bilstm': BiLSTMGNN
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")

    return models[model_name](**kwargs)
