"""
Task 2-2: MHA with KV Cache for autoregressive decoding

Demonstrates:
- past_key_values cache parameter in MHA forward()
- Streaming generation simulation: initial seq_len=10, then 5 steps of +1
- Prints KV cache shape each step to prove K/V grows from 10 → 15
  while Q seq_len stays 1 in the generation phase
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


KVCache = Tuple[torch.Tensor, torch.Tensor]   # (cached_K, cached_V)


class MHAWithKVCache(nn.Module):
    """Multi-Head Attention that supports an explicit KV Cache.

    Args:
        hidden_dim: total model dimension
        num_heads:  number of attention heads
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x:               (B, S_new, D)  — new tokens only (S_new=1 during generation)
            past_key_values: (cached_K, cached_V) each shaped (B, H, S_past, head_dim)
                             or None for the first call (prefill)

        Returns:
            out:             (B, S_new, D)
            new_kv:          updated (K, V) cache  — (B, H, S_past + S_new, head_dim)
        """
        # 1. Compute Q, K, V for the *new* tokens only
        Q_new = self._split_heads(self.W_q(x))   # (B, H, S_new, head_dim)
        K_new = self._split_heads(self.W_k(x))
        V_new = self._split_heads(self.W_v(x))

        # 2. Concatenate with cached K, V
        if past_key_values is not None:
            K_cached, V_cached = past_key_values
            K = torch.cat([K_cached, K_new], dim=2)   # (B, H, S_past+S_new, head_dim)
            V = torch.cat([V_cached, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        # 3. Attention: Q attends over the *full* K/V (past + current)
        scores      = torch.matmul(Q_new, K.transpose(-2, -1)) / self.scale  # (B,H,S_new,S_full)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out    = torch.matmul(attn_weights, V)                           # (B,H,S_new,head_dim)

        # 4. Merge heads and project
        out = self.W_o(self._merge_heads(attn_out))   # (B, S_new, D)

        # 5. Return output + updated cache
        new_kv: KVCache = (K, V)
        return out, new_kv


# ── Streaming generation simulation ──────────────────────────────────────────
def simulate_generation(
    model: MHAWithKVCache,
    batch_size: int = 1,
    hidden_dim: int = 64,
    prefill_len: int = 10,
    gen_steps:   int = 5,
):
    model.eval()
    torch.manual_seed(0)

    print("=" * 55)
    print("Step 0 — Prefill (initial sequence)")
    x_prefill = torch.randn(batch_size, prefill_len, hidden_dim)
    out, kv_cache = model(x_prefill, past_key_values=None)

    K, V = kv_cache
    print(f"  Input  shape : {x_prefill.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Cache K shape: {K.shape}   (B, H, seq, head_dim)")
    print(f"  Cache V shape: {V.shape}")

    for step in range(1, gen_steps + 1):
        print(f"\nStep {step} — Generate token {prefill_len + step}")

        # One new random token (simulates predicting the next token)
        x_new = torch.randn(batch_size, 1, hidden_dim)

        out, kv_cache = model(x_new, past_key_values=kv_cache)

        K, V = kv_cache
        print(f"  Q seq_len    : 1   (always 1 during generation)")
        print(f"  Cache K shape: {K.shape}   ← seq_len grew by 1")
        print(f"  Cache V shape: {V.shape}")

    print("=" * 55)
    print(f"\nFinal KV cache seq_len: {K.shape[2]}  (expected {prefill_len + gen_steps})")


if __name__ == "__main__":
    hidden_dim = 64
    num_heads  = 8

    model = MHAWithKVCache(hidden_dim=hidden_dim, num_heads=num_heads)

    simulate_generation(
        model,
        batch_size  = 1,
        hidden_dim  = hidden_dim,
        prefill_len = 10,
        gen_steps   = 5,
    )
