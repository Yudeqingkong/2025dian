"""
Task 2-3: Grouped Query Attention (GQA)

A single class GroupedQueryAttention unifies three architectures via num_kv_heads:
  - num_kv_heads == num_q_heads  →  MHA  (every Q head has its own K/V)
  - num_kv_heads == 1            →  MQA  (all Q heads share one K/V pair)
  - 1 < num_kv_heads < num_q_heads →  GQA  (groups of Q heads share one K/V pair)

Key idea: project K and V to a smaller head count, then expand (repeat_interleave)
to match Q before computing attention.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


KVCache = Tuple[torch.Tensor, torch.Tensor]


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention.

    Args:
        hidden_dim:   total model dimension D
        num_q_heads:  number of query heads (H_q)
        num_kv_heads: number of key/value heads (H_kv)
                      must divide num_q_heads evenly
        dropout:      attention dropout
    """

    def __init__(
        self,
        hidden_dim:   int,
        num_q_heads:  int,
        num_kv_heads: int,
        dropout:      float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_q_heads == 0,  "hidden_dim must be divisible by num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        self.hidden_dim   = hidden_dim
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups       = num_q_heads // num_kv_heads   # Q heads per KV head
        self.head_dim     = hidden_dim // num_q_heads
        self.scale        = math.sqrt(self.head_dim)

        # Q projects to full D; K and V project to the smaller KV dimension
        kv_dim = num_kv_heads * self.head_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, kv_dim,     bias=False)
        self.W_v = nn.Linear(hidden_dim, kv_dim,     bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _split_q(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) → (B, H_q, S, head_dim)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _split_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, kv_dim) → (B, H_kv, S, head_dim)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H_kv, S, head_dim) → (B, H_q, S, head_dim) by repeating each KV head."""
        # repeat_interleave duplicates each head `groups` times along dim=1
        return x.repeat_interleave(self.groups, dim=1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H_q, S, head_dim) → (B, S, D)"""
        B, H, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x:               (B, S_new, D)
            past_key_values: optional (K_cache, V_cache) each (B, H_kv, S_past, head_dim)
            mask:            optional causal/padding mask

        Returns:
            out:     (B, S_new, D)
            new_kv:  updated (K, V) cache at H_kv resolution
        """
        # Project
        Q   = self._split_q(self.W_q(x))    # (B, H_q,  S_new, head_dim)
        K_new = self._split_kv(self.W_k(x)) # (B, H_kv, S_new, head_dim)
        V_new = self._split_kv(self.W_v(x)) # (B, H_kv, S_new, head_dim)

        # Append to cache (cache is stored at H_kv resolution to save memory)
        if past_key_values is not None:
            K_cached, V_cached = past_key_values
            K = torch.cat([K_cached, K_new], dim=2)
            V = torch.cat([V_cached, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        # Expand K/V from H_kv → H_q for the attention computation
        K_expanded = self._expand_kv(K)   # (B, H_q, S_full, head_dim)
        V_expanded = self._expand_kv(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V_expanded)   # (B, H_q, S_new, head_dim)

        out = self.W_o(self._merge_heads(attn_out))          # (B, S_new, D)

        return out, (K, V)   # cache stays at H_kv size


# ── Verification ──────────────────────────────────────────────────────────────
def verify(name: str, num_q_heads: int, num_kv_heads: int):
    torch.manual_seed(0)
    B, S, D = 2, 8, 64

    model = GroupedQueryAttention(
        hidden_dim=D, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
    )
    x = torch.randn(B, S, D)
    out, (K, V) = model(x)

    kv_params = sum(p.numel() for p in list(model.W_k.parameters()) + list(model.W_v.parameters()))
    print(
        f"{name:<6}  num_kv_heads={num_kv_heads}  "
        f"input={tuple(x.shape)}  output={tuple(out.shape)}  "
        f"K shape={tuple(K.shape)}  KV params={kv_params}"
    )
    assert out.shape == x.shape


if __name__ == "__main__":
    print("Architecture verification")
    print("-" * 70)
    verify("MHA",  num_q_heads=8, num_kv_heads=8)   # standard MHA
    verify("GQA",  num_q_heads=8, num_kv_heads=2)   # grouped (4 Q per KV head)
    verify("MQA",  num_q_heads=8, num_kv_heads=1)   # multi-query
    print("\nAll shape checks passed.")

    # ── KV memory comparison ──────────────────────────────────────────────────
    print("\nKV Cache memory footprint (B=1, S=1024, D=512, H_q=8):")
    print("-" * 50)
    D, S, H_q = 512, 1024, 8
    for name, H_kv in [("MHA", 8), ("GQA-4", 4), ("GQA-2", 2), ("MQA", 1)]:
        head_dim   = D // H_q
        kv_entries = 2 * H_kv * S * head_dim        # K + V
        ratio      = H_kv / H_q
        print(f"  {name:<8}  H_kv={H_kv}  KV entries={kv_entries:>8,}  ratio vs MHA={ratio:.2f}")
