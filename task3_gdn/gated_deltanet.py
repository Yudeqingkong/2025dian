"""
Task 3-1 / 3-2 / 3-3: Gated DeltaNet (GDN)

Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
       https://arxiv.org/abs/2412.06464

Core update rule (recurrent mode):
    S_t = S_{t-1} @ (α_t (I − β_t k_t k_t^T)) + β_t v_t k_t^T
    o_t = S_t @ q_t

GDN Block architecture:
    Input → q,k: Linear → CausalConv → SiLU → L2Norm
          → v:   Linear → CausalConv → SiLU
          → α:   Linear → Sigmoid
          → β:   Linear → Sigmoid
          → Gated Delta Rule (recurrent)
          → Zero-Centered RMSNorm ⊗ OutputGate(SiLU)
          → Linear → Output

This file provides:
    - ZeroCenteredRMSNorm
    - GatedDeltaNet        (core layer)
    - GDNBlock             (GDN + LayerNorm + MLP with residual connections)
    - PatchEmbedding       (image → patch sequence)
    - GDNClassifier        (full model for Fashion-MNIST)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with learnable gain initialized to 0 → starts as pure normalization.

    Formula: x / RMS(x) * (1 + γ),  where γ is initialized to 0.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * (1.0 + self.gamma)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Layer: Gated DeltaNet
# ═══════════════════════════════════════════════════════════════════════════════

class GatedDeltaNet(nn.Module):
    """Gated DeltaNet core layer (recurrent mode).

    Args:
        hidden_dim: model dimension D
        num_heads:  number of heads H
        head_dim:   per-head dimension d  (default D // H)
        conv_size:  short causal convolution kernel size (default 4)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        conv_size: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads
        self.inner_dim = num_heads * self.head_dim
        self.conv_size = conv_size

        # ── Linear projections ────────────────────────────────────────────────
        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)

        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)   # decay gate
        self.beta_proj  = nn.Linear(hidden_dim, num_heads, bias=True)   # input gate

        self.gate_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)  # output gate
        self.out_proj  = nn.Linear(self.inner_dim, hidden_dim, bias=False)

        # ── Depthwise causal convolutions ─────────────────────────────────────
        self.q_conv = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=conv_size, padding=0, groups=self.inner_dim, bias=True,
        )
        self.k_conv = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=conv_size, padding=0, groups=self.inner_dim, bias=True,
        )
        self.v_conv = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=conv_size, padding=0, groups=self.inner_dim, bias=True,
        )

        # ── Output normalization ──────────────────────────────────────────────
        self.norm = ZeroCenteredRMSNorm(self.inner_dim)

        self._init_weights()

    def _init_weights(self):
        # Bias α high → sigmoid ≈ 0.88 → model remembers past by default
        nn.init.constant_(self.alpha_proj.bias, 2.0)
        nn.init.constant_(self.beta_proj.bias, 0.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply depthwise causal 1-D convolution.  (B,T,C) → (B,T,C)"""
        x = x.transpose(1, 2)                           # (B, C, T)
        x = F.pad(x, (self.conv_size - 1, 0))           # left-pad for causality
        x = conv(x)
        return x.transpose(1, 2)                         # (B, T, C)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # ── 1. Project + short conv + activation ─────────────────────────────
        q = F.silu(self._causal_conv(self.q_proj(x), self.q_conv))   # (B, T, inner)
        k = F.silu(self._causal_conv(self.k_proj(x), self.k_conv))
        v = F.silu(self._causal_conv(self.v_proj(x), self.v_conv))

        # ── 2. Reshape to heads & L2-normalize q, k ─────────────────────────
        q = F.normalize(q.view(B, T, H, d), p=2, dim=-1)            # (B, T, H, d)
        k = F.normalize(k.view(B, T, H, d), p=2, dim=-1)
        v = v.view(B, T, H, d)

        # ── 3. Scalar gates (per head) ───────────────────────────────────────
        alpha = torch.sigmoid(self.alpha_proj(x))                     # (B, T, H)
        beta  = torch.sigmoid(self.beta_proj(x))                      # (B, T, H)

        # ── 4. Output gate ────────────────────────────────────────────────────
        gate = F.silu(self.gate_proj(x))                              # (B, T, inner)

        # ── 5. Recurrent delta-rule computation ──────────────────────────────
        # State matrix S: (B, H, d_v, d_k)  (here d_v = d_k = d)
        S = x.new_zeros(B, H, d, d)
        I_d = torch.eye(d, device=x.device, dtype=x.dtype)           # (d, d)

        outputs: list[torch.Tensor] = []
        for t in range(T):
            q_t = q[:, t]                                             # (B, H, d)
            k_t = k[:, t]
            v_t = v[:, t]
            a_t = alpha[:, t, :, None, None]                          # (B, H, 1, 1)
            b_t = beta[:, t, :, None, None]

            # Outer products  (B, H, d, d)
            kk_T = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)             # k k^T
            vk_T = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)             # v k^T

            # S_t = S_{t-1} @ [α(I − β k k^T)]  +  β v k^T
            decay = a_t * (I_d - b_t * kk_T)                         # (B, H, d, d)
            S = torch.matmul(S, decay) + b_t * vk_T

            # o_t = S_t @ q_t
            o_t = torch.einsum("bhde,bhe->bhd", S, q_t)              # (B, H, d)
            outputs.append(o_t)

        # ── 6. Assemble, norm, gate, project ─────────────────────────────────
        output = torch.stack(outputs, dim=1)                          # (B, T, H, d)
        output = output.reshape(B, T, self.inner_dim)
        output = self.norm(output) * gate                             # norm ⊗ gate
        output = self.out_proj(output)                                # (B, T, D)

        return output

    # ── Chunkwise 并行形式 ─────────────────────────────────────────────────

    def forward_chunkwise(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """Chunkwise 并行形式的前向传播。

        核心思想（前缀和/Parallel Prefix Scan）：
        - 将长度为 T 的序列分成 T/C 个大小为 C 的 chunk
        - 将递推 S_t = S_{t-1} @ D_t + U_t 视为关联运算 (D, U)
          其中 (D_a, U_a) ⊕ (D_b, U_b) = (D_a @ D_b, U_a @ D_b + U_b)
        - chunk 内：利用前缀扫描并行计算所有位置的累积 (cumD, cumU)，
          再一次性 batch matmul 算出所有位置的输出
        - chunk 间：串行传递状态，但仅需 T/C 步

        串行深度从 O(T) 降到 O(T/C + C)，计算吞吐量显著提升。

        Args:
            x:          (batch, seq_len, hidden_dim)
            chunk_size: chunk 大小 C，默认 8
        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.shape
        H, d = self.num_heads, self.head_dim
        C = chunk_size

        # ── 0. 补齐到 chunk_size 的整数倍 ──────────────────────────────────
        pad_len = (C - T % C) % C
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_pad = T + pad_len
        num_chunks = T_pad // C

        # ── 1. 投影 + 短卷积 + 激活（全序列并行） ──────────────────────────
        q = F.silu(self._causal_conv(self.q_proj(x), self.q_conv))
        k = F.silu(self._causal_conv(self.k_proj(x), self.k_conv))
        v = F.silu(self._causal_conv(self.v_proj(x), self.v_conv))

        q = F.normalize(q.view(B, T_pad, H, d), p=2, dim=-1)
        k = F.normalize(k.view(B, T_pad, H, d), p=2, dim=-1)
        v = v.view(B, T_pad, H, d)

        alpha = torch.sigmoid(self.alpha_proj(x))       # (B, T_pad, H)
        beta  = torch.sigmoid(self.beta_proj(x))
        gate  = F.silu(self.gate_proj(x))                # (B, T_pad, inner)

        # ── 2. 并行计算所有位置的 D_t 和 U_t ──────────────────────────────
        #   D_t = α_t (I − β_t k_t k_t^T)    衰减矩阵
        #   U_t = β_t v_t k_t^T               更新矩阵
        I_d = torch.eye(d, device=x.device, dtype=x.dtype)

        kk_T = k.unsqueeze(-1) * k.unsqueeze(-2)        # (B, T_pad, H, d, d)
        vk_T = v.unsqueeze(-1) * k.unsqueeze(-2)

        a_exp = alpha[..., None, None]                    # (B, T_pad, H, 1, 1)
        b_exp = beta[..., None, None]

        D_all = a_exp * (I_d - b_exp * kk_T)             # (B, T_pad, H, d, d)
        U_all = b_exp * vk_T

        # 重塑为 chunk 形状
        D_chunks = D_all.view(B, num_chunks, C, H, d, d)
        U_chunks = U_all.view(B, num_chunks, C, H, d, d)
        q_chunks = q.view(B, num_chunks, C, H, d)

        # ── 3. 逐 chunk 处理：内部前缀扫描 + 并行输出 ──────────────────────
        S = x.new_zeros(B, H, d, d)                       # chunk 间传递的状态
        all_outputs: list[torch.Tensor] = []

        for c in range(num_chunks):
            D_c = D_chunks[:, c]   # (B, C, H, d, d)
            U_c = U_chunks[:, c]
            q_c = q_chunks[:, c]   # (B, C, H, d)

            # ── 前缀扫描（Prefix Scan）：计算 cumD[r] 和 cumU[r] ──────
            #   cumD[r] = D_0 @ D_1 @ ... @ D_r
            #   cumU[r] = cumU[r-1] @ D_r + U_r,  cumU[-1] = 0
            #   => 状态 S_r = S_init @ cumD[r] + cumU[r]
            #
            #   此处用 O(C) 循环实现（C 很小，通常 7~16），
            #   也可进一步用 tree-reduction 优化到 O(log C) 深度。
            cumD_list: list[torch.Tensor] = []
            cumU_list: list[torch.Tensor] = []
            cD = I_d.unsqueeze(0).unsqueeze(0).expand(B, H, d, d).clone()
            cU = x.new_zeros(B, H, d, d)

            for r in range(C):
                cD = torch.matmul(cD, D_c[:, r])          # (B, H, d, d)
                cU = torch.matmul(cU, D_c[:, r]) + U_c[:, r]
                cumD_list.append(cD)
                cumU_list.append(cU)

            cumD = torch.stack(cumD_list, dim=1)           # (B, C, H, d, d)
            cumU = torch.stack(cumU_list, dim=1)

            # ── 并行计算 chunk 内所有位置的输出 ─────────────────────────
            #   S_r = S_init @ cumD[r] + cumU[r]
            #   o_r = S_r @ q_r
            S_init = S.unsqueeze(1)                        # (B, 1, H, d, d)
            states = torch.matmul(S_init, cumD) + cumU     # (B, C, H, d, d)

            # batch einsum: (B, C, H, d, d) × (B, C, H, d) → (B, C, H, d)
            o_c = torch.einsum("bchde,bche->bchd", states, q_c)
            all_outputs.append(o_c)

            # 更新 chunk 间状态为本 chunk 最后一个位置的状态
            S = states[:, -1]                              # (B, H, d, d)

        # ── 4. 拼接、去 padding、归一化、门控、投影 ────────────────────────
        output = torch.cat(all_outputs, dim=1)             # (B, T_pad, H, d)
        if pad_len > 0:
            output = output[:, :T]
            gate = gate[:, :T]

        output = output.reshape(B, T, self.inner_dim)
        output = self.norm(output) * gate
        output = self.out_proj(output)

        return output


# ═══════════════════════════════════════════════════════════════════════════════
# Feed-Forward / MLP
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Two-layer feed-forward: Linear → GELU → Linear."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# GDN Block  (pre-norm residual)
# ═══════════════════════════════════════════════════════════════════════════════

class GDNBlock(nn.Module):
    """Pre-norm block: LN → GDN → +residual → LN → MLP → +residual."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_expansion: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gdn   = GatedDeltaNet(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp   = MLP(hidden_dim, mlp_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gdn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Patch Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Reshape an image into a flat sequence of patch embeddings.

    (B, C, H, W) → (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C, H//p, p, W//p, p) → (B, H//p * W//p, C*p*p)
        x = x.unfold(2, p, p).unfold(3, p, p)           # (B, C, nH, nW, p, p)
        x = x.contiguous().view(B, -1, C * p * p)        # (B, num_patches, patch_dim)
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class GDNClassifier(nn.Module):
    """GDN-based image classifier for Fashion-MNIST (or similar).

    Pipeline:
        PatchEmbed + Learnable PosEmbed
        → N × GDNBlock
        → LayerNorm → Global Avg Pool → Linear head
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_expansion: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional encoding  (策略优化 3-3: 位置编码)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [GDNBlock(hidden_dim, num_heads, mlp_expansion) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 28, 28)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.patch_embed(x) + self.pos_embed          # (B, num_patches, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)                                  # global average pooling
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 测试核心层 ---
    print("=== GatedDeltaNet 核心层 ===")
    layer = GatedDeltaNet(hidden_dim=64, num_heads=4).to(device)
    x = torch.randn(2, 16, 64, device=device)
    y = layer(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"
    print("Shape check passed.\n")

    # --- 测试 Chunkwise 并行形式 ---
    print("=== Chunkwise 并行 vs Recurrent 等效性验证 ===")
    layer.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 49, 64, device=device)    # 49 tokens = 7×7 patches
        out_recurrent  = layer(x_test)                      # 纯循环模式
        out_chunkwise  = layer.forward_chunkwise(x_test, chunk_size=7)  # Chunkwise

        max_diff = (out_recurrent - out_chunkwise).abs().max().item()
        print(f"Recurrent  output shape: {out_recurrent.shape}")
        print(f"Chunkwise  output shape: {out_chunkwise.shape}")
        print(f"Max absolute difference: {max_diff:.2e}")
        assert max_diff < 1e-4, f"输出不一致! max_diff={max_diff}"
        print("等效性验证通过: Recurrent ≈ Chunkwise\n")

    # --- 测试完整分类器 ---
    print("=== GDNClassifier (Fashion-MNIST) ===")
    model = GDNClassifier(
        img_size=28, patch_size=4, hidden_dim=64,
        num_heads=4, num_layers=3, num_classes=10,
    ).to(device)
    imgs = torch.randn(4, 1, 28, 28, device=device)
    logits = model(imgs)
    print(f"Input:  {imgs.shape}")
    print(f"Output: {logits.shape}")
    assert logits.shape == (4, 10)
    print("Shape check passed.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
