"""
Task 2-1: Standard Multi-Head Attention (MHA)

Rules:
- Only nn.Linear and basic matrix ops (no nn.MultiheadAttention)
- Output shape must equal input shape: (batch, seq_len, hidden_dim)
"""

import math
import torch
import torch.nn as nn


class StandardMHA(nn.Module):
    """Standard Multi-Head Attention.

    Args:
        hidden_dim: total model dimension (must be divisible by num_heads)
        num_heads:  number of attention heads
        dropout:    attention dropout probability
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        # Four projection matrices: Q, K, V, and output
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) → (B, H, S, head_dim)"""
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.transpose(1, 2)   # (B, H, S, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, S, head_dim) → (B, S, D)"""
        B, H, S, _ = x.shape
        x = x.transpose(1, 2).contiguous()   # (B, S, H, head_dim)
        return x.view(B, S, self.hidden_dim)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Q, K, V: (B, H, S, head_dim)
        Returns: (B, H, S, head_dim)
        """
        # (B, H, S_q, S_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)   # (B, H, S_q, S_k)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, S_q, head_dim)
        return torch.matmul(attn_weights, V)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch_size, seq_len, hidden_dim)
            mask: optional boolean mask (batch_size, 1, seq_len, seq_len)

        Returns:
            out:  (batch_size, seq_len, hidden_dim)  — same shape as input
        """
        Q = self._split_heads(self.W_q(x))   # (B, H, S, head_dim)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)  # (B, H, S, head_dim)
        merged   = self._merge_heads(attn_out)                        # (B, S, D)
        out      = self.W_o(merged)                                   # (B, S, D)

        return out


# ── Verification ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len    = 8
    hidden_dim = 64
    num_heads  = 8

    x   = torch.randn(batch_size, seq_len, hidden_dim)
    mha = StandardMHA(hidden_dim=hidden_dim, num_heads=num_heads)

    out = mha(x)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("Shape check passed: output == input shape")
