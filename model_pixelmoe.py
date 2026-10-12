# --------------------------------------------------------
# References:
# JiT: https://github.com/LTH14/JiT
# FLUX: https://github.com/black-forest-labs/flux
# Megatron-LM (weight initialization): https://github.com/NVIDIA/Megatron-LM
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from deepseek_moe import (
    DeepSeekMoEFFN,
    DeepSeekMoEGate,
    DeepSeekSharedExpert,
)

from typing import List, Optional
from einops import rearrange
from util.model_util import get_2d_sincos_pos_embed, RMSNorm

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except Exception:
    flash_attn_varlen_func = None


# Helpers

def modulate(x, shift: torch.Tensor, scale: torch.Tensor):
    # Apply scale and shift to packed tokens; inputs and output: [1, total_tokens, D].
    return x * (1 + scale) + shift


def expand_segment_embeddings(embed: torch.Tensor, hidden_length: List[int], batch_size: int):
    """
    Repeat each image's conditioning over its packed tokens.

    Args
        embed (torch.Tensor):
            Conditioning embeddings with shape [num_images, D].
        hidden_length (List[int]):
            Token count per image, including context tokens when present.
        batch_size (int):
            1 for packed tokens.
    Returns
        token embeddings with shape [1, sum(hidden_length), D].
    """
    num_segments = len(hidden_length)
    lengths = torch.as_tensor(hidden_length, device=embed.device, dtype=torch.long)
    embed = embed.reshape(batch_size, num_segments, -1).transpose(0, 1)  # [num_segments, 1, D]
    return torch.repeat_interleave(embed, lengths, dim=0).transpose(0, 1).contiguous()  # [1, sum(hidden_length), D]


@torch._dynamo.disable
def split_ragged_context_image_tokens(mlp_input, lengths, context_len):
    """
    Split packed tokens into context and image groups.

    Args
        mlp_input (torch.Tensor):
            Packed tokens with shape [1, sum(lengths), D].
        lengths (List[int]):
            Token count per image, including context tokens.
        context_len (int):
            Context token count per image.
    Returns
        context tokens [1, num_images * context_len, D],
        image tokens [1, sum(image_lengths), D],
        image token counts.
    """
    segments = torch.split(mlp_input, lengths, dim=1)  # Each segment: [1, lengths[i], D].
    context_input = torch.cat(
        [segment[:, :context_len] for segment in segments],
        dim=1,
    )  # [1, num_images * context_len, D]
    image_lengths = [length - context_len for length in lengths]  # Image token count per image.
    image_input = torch.cat(
        [segment[:, context_len:] for segment in segments],
        dim=1,
    )  # [1, sum(image_lengths), D]
    return context_input, image_input, image_lengths


@torch._dynamo.disable
def merge_ragged_context_image_tokens(
    context_output,
    image_output,
    image_lengths,
    context_len,
):
    """
    Reassemble packed tokens in each image's context-then-image order.

    Args
        context_output (torch.Tensor):
            Context tokens with shape [1, num_images * context_len, D].
        image_output (torch.Tensor):
            Image tokens with shape [1, sum(image_lengths), D].
        image_lengths (List[int]):
            Image token count per image, excluding context tokens.
        context_len (int):
            Context token count per image.
    Returns
        packed tokens [1, num_images * context_len + sum(image_lengths), D].
    """
    context_segments = torch.split(
        context_output,
        [context_len] * len(image_lengths),
        dim=1,
    )  # Each segment: [1, context_len, D].
    image_segments = torch.split(image_output, image_lengths, dim=1)  # Each segment: [1, image_lengths[i], D].
    return torch.cat(
        [
            torch.cat([context_segment, image_segment], dim=1)  # [1, context_len + image_lengths[i], D]
            for context_segment, image_segment in zip(
                context_segments,
                image_segments,
            )
        ],
        dim=1,
    )  # [1, num_images * context_len + sum(image_lengths), D]


# RoPE

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Build 2x2 rotation matrices for one position axis.

    Args
        pos (torch.Tensor):
            Positions with shape [B, L].
        dim (int):
            Even feature dimension for this axis.
        theta (int):
            Frequency base.
    Returns
        FP32 rotation matrices with shape [B, L, dim // 2, 2, 2].
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size = pos.shape[0]
    out = torch.einsum("...n,d->...nd", pos, omega)  # [B, L, dim // 2]
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    # Rotation matrix: [[cos, -sin], [sin, cos]].
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()  # [B, L, dim // 2, 2, 2]


def apply_rope(xq, xk, freqs_cis):
    """
    Rotate query and key channel pairs in FP32 (B=1 for packed tokens).

    Args
        xq (torch.Tensor):
            Query with shape [B, L, heads, head_dim].
        xk (torch.Tensor):
            Key with shape [B, L, heads, head_dim].
        freqs_cis (torch.Tensor):
            Rotation matrices [B, L, 1, head_dim // 2, 2, 2], shared across heads.
    Returns
        rotated query and key with their original shapes and dtypes.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)  # [B, L, heads, head_dim // 2, 1, 2]
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]  # [B, L, heads, head_dim // 2, 2]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """
    Build RoPE matrices for each position axis and concatenate them.

    Args
        theta (int):
            Frequency base.
        axes_dim (List[int]):
            Feature dimension per axis; height and width in this model.
    """
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Embed each token's position axes as RoPE matrices.

        Args
            ids (torch.Tensor):
                Positions [B, L, num_axes]; B=1 and num_axes=2 for image tokens.
        Returns
            torch.Tensor: FP32 matrices [B, L, 1, sum(axes_dim) // 2, 2, 2], shared across heads.
        """
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )  # [B, L, sum(axes_dim) // 2, 2, 2]
        return emb.unsqueeze(2)  # [B, L, 1, sum(axes_dim) // 2, 2, 2]


# Condition embedders

class TimestepEmbedder(nn.Module):
    """
    Embeds global timestep t into a vector; stage-relative sigma
    is used by the loss and solver, and is not passed to this embedder.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        max_period: int = 10000,
    ):
        """
        Create sinusoidal embeddings of global timesteps.

        Args
            timesteps (torch.Tensor):
                a 1-D tensor of N global timestep values, one per image.
            embedding_dim (int):
                the dimension of the output.
            max_period (int):
                Sets the largest period in the frequency scale.
        Returns
            timestep embeddings with shape [N, embedding_dim].
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor."

        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        ) # [Dim // 2]
        exponent = exponent / (half_dim - 1)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :] # [B, Dim // 2]

        # Concatenate cosine and sine embeddings.
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1) # [B, Dim]

        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embed class labels; index num_classes represents the unconditional label.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


# Attention

@torch._dynamo.disable
def build_flash_cu_seqlens(hidden_length, device):
    """
    Build cumulative token offsets for FlashAttention in eager mode.

    Args
        hidden_length (List[int]):
            Token count per image in packed order.
        device (torch.device):
            Device for the offsets tensor.
    Returns
        offsets with shape [num_images + 1], starting at zero.
    """
    offsets = [0]
    total = 0
    for length in hidden_length:
        length = int(length)
        total += length
        offsets.append(total)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


class VarlenFlashAttn(nn.Module):
    @torch._dynamo.disable
    def forward(self, query, key, value, cu_seqlens, max_seqlen, scale=None):
        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
        )


class VarlenFlashSelfAttnSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.flash_attn = VarlenFlashAttn()

    def forward(
        self,
        query,
        key,
        value,
        scale,
        hidden_length=None,
        image_rotary_emb=None,
    ):
        if flash_attn_varlen_func is None:
            raise ImportError("flash-attn is required when use_flash_attn=True")

        # Stack Q/K/V and apply RoPE to Q/K only.
        qkv = torch.stack([query, key, value], dim=2)
        if image_rotary_emb is not None:
            qkv[:, :, 0], qkv[:, :, 1] = apply_rope(
                qkv[:, :, 0],
                qkv[:, :, 1],
                image_rotary_emb,
            )

        qkv = rearrange(qkv, "b s ... -> (b s) ...")  # [total_tokens, 3, heads, head_dim]
        query, key, value = qkv.unbind(1)

        # Token offsets keep attention separate for each image in the packed sequence.
        max_seqlen_q = max(hidden_length)
        cu_seqlens_q = build_flash_cu_seqlens(
            hidden_length,
            query.device,
        )
        output = self.flash_attn(query, key, value, cu_seqlens_q, max_seqlen_q, scale)
        return output.reshape(1, -1, output.shape[-2], output.shape[-1]).flatten(2, 3)  # [1, total_tokens, heads * head_dim]


class VarlenSelfAttnSingle:
    def __init__(self):
        pass

    def __call__(
        self,
        query,
        key,
        value,
        hidden_length=None,
        image_rotary_emb=None,
    ):

        # Stack Q/K/V and apply RoPE to Q/K only.
        qkv = torch.stack([query, key, value], dim=2)  # [1, total_tokens, 3, heads, head_dim]
        if image_rotary_emb is not None:
            qkv[:, :, 0], qkv[:, :, 1] = apply_rope(
                qkv[:, :, 0],
                qkv[:, :, 1],
                image_rotary_emb,
            )

        i_sum = 0
        output_hidden_list = []
        # Compute attention within each image's token segment.
        for length in hidden_length:
            qkv_tokens = qkv[:, i_sum:i_sum + length]  # [1, length, 3, heads, head_dim]

            query, key, value = qkv_tokens.unbind(2)
            # SDPA expects Q/K/V in [1, heads, length, head_dim].
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

            stage_hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            stage_hidden_states = stage_hidden_states.transpose(1, 2).flatten(2, 3)  # [1, length, heads * head_dim]
            output_hidden_list.append(stage_hidden_states)
            i_sum += length

        return torch.cat(output_hidden_list, dim=1)  # [1, total_tokens, heads * head_dim]


class FLUXAttention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_flash_attn=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=False),
            nn.Dropout(dropout),
        ])
        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            self.varlen_flash_attn = VarlenFlashSelfAttnSingle()
        else:
            self.varlen_attn = VarlenSelfAttnSingle()

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_length: List = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Project packed tokens to Q/K/V: [1, total_tokens, inner_dim].
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        # Split attention heads: [1, total_tokens, heads, head_dim].
        query = query.view(query.shape[0], -1, self.heads, head_dim)
        key = key.view(key.shape[0], -1, self.heads, head_dim)
        value = value.view(value.shape[0], -1, self.heads, head_dim)

        # Apply RMSNorm to each head's Q/K features.
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Compute per-image attention and merge heads: [1, total_tokens, inner_dim].
        if self.use_flash_attn:
            hidden_states = self.varlen_flash_attn(
                query, key, value, 
                self.scale, hidden_length, 
                image_rotary_emb,
            )
        else:
            hidden_states = self.varlen_attn(
                query, key, value, 
                hidden_length, 
                image_rotary_emb,
            )

        # Project back to the input width, then apply dropout.
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


# Model components

class BottleneckPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class PixelMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        flux_attn_proj_dropout=False,
        use_flash_attn=False,
        mlp_ratio=4.0,
        proj_drop=0.0,
        use_moe=False,
        moe_num_experts=8,
        moe_top_k=2,
        moe_routed_inter_dim=None,
        moe_shared_inter_dim=None,
        moe_route_scale=1.0,
        context_token_count=0,
        moe_router_bias_update_rate=1e-3,
        moe_num_layers=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        attention_head_dim = hidden_size // num_heads
        self.attn = FLUXAttention(
            query_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_heads,
            eps=1e-6,
            use_flash_attn=use_flash_attn,
            dropout=proj_drop if flux_attn_proj_dropout else 0.0,
        )

        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_moe:
            self.mlp = DeepSeekMoEFFN(
                hidden_size,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                routed_inter_dim=moe_routed_inter_dim,
                shared_inter_dim=moe_shared_inter_dim,
                route_scale=moe_route_scale,
                router_bias_update_rate=moe_router_bias_update_rate,
                drop=proj_drop,
                num_layers=moe_num_layers,
            )
        else:
            self.mlp = SwiGLUFFN(
                hidden_size,
                mlp_hidden_dim,
                drop=proj_drop,
            )
        self.class_token_mlp = (
            SwiGLUFFN(
                hidden_size,
                mlp_hidden_dim,
                drop=proj_drop,
            )
            if use_moe and context_token_count > 0
            else None
        )
        self.context_token_count = int(context_token_count)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def _forward_mlp(
        self,
        mlp_input,
        hidden_length=None,
    ):
        # Use one FFN when no separate class-token FFN is needed.
        if self.class_token_mlp is None:
            return self.mlp(
                mlp_input,
            )

        context_len = self.context_token_count
        # Split packed context/image tokens.
        lengths = [int(length) for length in hidden_length]
        if any(length <= context_len for length in lengths):
            raise ValueError(
                "Every ragged segment must contain class-context and image tokens; "
                f"got lengths={lengths}, context_token_count={context_len}"
            )
        context_input, image_input, image_lengths = split_ragged_context_image_tokens(
            mlp_input,
            lengths,
            context_len,
        )

        context_output = self.class_token_mlp(context_input)
        image_output = self.mlp(
            image_input,
        )
        return merge_ragged_context_image_tokens(
            context_output,
            image_output,
            image_lengths,
            context_len,
        )

    @torch.compile
    def forward(
        self,
        x,
        c,
        hidden_length=None,
        image_rotary_emb=None,
    ):
        # Expand each image's AdaLN conditioning to its packed tokens.
        embed = self.adaLN_modulation(c)
        batch_emb = expand_segment_embeddings(embed, hidden_length, x.shape[0])  # [1, total_tokens, 6 * D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = batch_emb.chunk(6, dim=-1)

        # Attention with AdaLN modulation and a gated residual.
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(
            hidden_states=attn_input,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )
        x = x + gate_msa * attn_output

        # FFN with AdaLN modulation and a gated residual.
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self._forward_mlp(
            mlp_input,
            hidden_length=hidden_length,
        )
        x = x + gate_mlp * mlp_output
        return x


class PixelMoEDDTBlock(nn.Module):
    """
    Wide DDT head block with token-wise AdaLN conditioning of shape [B, L, D].
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        use_flash_attn=False,
        mlp_ratio=4.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        attention_head_dim = hidden_size // num_heads
        self.attn = FLUXAttention(
            query_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_heads,
            eps=1e-6,
            use_flash_attn=use_flash_attn,
        )

        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(
            hidden_size,
            mlp_hidden_dim,
            drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(
        self,
        x,
        c,
        hidden_length=None,
        image_rotary_emb=None,
    ):
        # Per-token AdaLN shift, scale, and gates; c: [1, total_tokens, D].
        embed = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = embed.chunk(6, dim=-1)

        # Attention with AdaLN modulation and a gated residual.
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(
            hidden_states=attn_input,
            hidden_length=hidden_length,
            image_rotary_emb=image_rotary_emb,
        )
        x = x + attn_output * gate_msa

        # Dense SwiGLU FFN with AdaLN modulation and a gated residual.
        mlp_output = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + mlp_output * gate_mlp
        return x


class DDTFinalLayer(nn.Module):
    """
    Token-wise conditioned final layer used by PixelMoE.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        # Apply per-token AdaLN conditioning before the output projection.
        embed = self.adaLN_modulation(c)
        shift, scale = embed.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # [1, total_tokens, patch_size * patch_size * out_channels]
        return x


# PixelMoE

class PixelMoE(nn.Module):
    """PixelMoE body with a wide DDT head and FLUX variable-length attention."""
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        flux_attn_proj_dropout=False,
        use_flash_attn=False,
        ddt_head_dim=2048,
        ddt_head_depth=2,
        ddt_head_num_heads=16,
        moe_num_experts=8,
        moe_top_k=2,
        moe_routed_inter_dim=None,
        moe_shared_inter_dim=None,
        moe_route_scale=1.0,
        moe_n_dense_layers=1,
        moe_router_bias_update_rate=1e-3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.use_flash_attn = use_flash_attn
        # Base initialization std is 0.02; residual outputs use body-depth scaling.
        self.megatron_init_std = 0.02
        self.megatron_init_num_layers = depth

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.in_context_posemb, std=.02)

        self.hierarchical_pos_embed_cache = {}

        attention_head_dim = hidden_size // num_heads
        self.hierarchical_rope_embed = EmbedND(
            theta=10000,
            axes_dim=[attention_head_dim // 2, attention_head_dim // 2],
        )

        # transformer
        self.blocks = nn.ModuleList([
            PixelMoEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_flash_attn=use_flash_attn, 
                    flux_attn_proj_dropout=flux_attn_proj_dropout,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    use_moe=i >= moe_n_dense_layers,
                    context_token_count=in_context_len if i >= in_context_start else 0,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_routed_inter_dim=moe_routed_inter_dim,
                    moe_shared_inter_dim=moe_shared_inter_dim,
                    moe_route_scale=moe_route_scale,
                    moe_router_bias_update_rate=moe_router_bias_update_rate,
                    moe_num_layers=self.megatron_init_num_layers,
                    )
            for i in range(depth)
        ])

        self.initialize_weights()

        self.ddt_x_embedder = BottleneckPatchEmbed(
            self.input_size,
            self.patch_size,
            self.in_channels,
            ddt_head_dim,
            ddt_head_dim,
        )
        self.ddt_s_projector = nn.Linear(self.hidden_size, ddt_head_dim)
        self.ddt_blocks = nn.ModuleList([
            PixelMoEDDTBlock(
                ddt_head_dim,
                ddt_head_num_heads,
                mlp_ratio=4.0,
                use_flash_attn=self.use_flash_attn,
            )
            for _ in range(ddt_head_depth)
        ])
        self.ddt_final_layer = DDTFinalLayer(ddt_head_dim, self.patch_size, self.out_channels)

        attention_head_dim = ddt_head_dim // ddt_head_num_heads
        self.ddt_hierarchical_rope_embed = EmbedND(
            theta=10000,
            axes_dim=[attention_head_dim // 2, attention_head_dim // 2],
        )
        self.initialize_ddt_weights()

    def initialize_weights(self):
        self._initialize_megatron_linear_modules(self)

        # Initialize (and freeze) pos_embed using sinusoidal embeddings:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self._initialize_megatron_patch_embed(self.x_embedder)
        nn.init.normal_(self.y_embedder.embedding_table.weight, mean=0.0, std=self.megatron_init_std)
        nn.init.normal_(self.in_context_posemb, mean=0.0, std=self.megatron_init_std)
        self._initialize_megatron_output_projections(self)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def initialize_ddt_weights(self):
        self._initialize_megatron_linear_modules(self.ddt_s_projector)
        self._initialize_megatron_linear_modules(self.ddt_blocks)
        self._initialize_megatron_patch_embed(self.ddt_x_embedder)
        self._initialize_megatron_output_projections(self.ddt_blocks)

        for block in self.ddt_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.ddt_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.ddt_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.ddt_final_layer.linear.weight, 0)
        nn.init.constant_(self.ddt_final_layer.linear.bias, 0)

    @property
    def megatron_output_init_std(self):
        return self.megatron_init_std / math.sqrt(2.0 * self.megatron_init_num_layers)

    def _initialize_megatron_linear_modules(self, root):
        """
        Initialize ordinary projections with the Megatron base std.
        """
        def _init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.megatron_init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, DeepSeekMoEGate):
                nn.init.normal_(module.weight, mean=0.0, std=self.megatron_init_std)

        root.apply(_init_linear)

    def _initialize_megatron_patch_embed(self, patch_embed):
        for projection in (patch_embed.proj1, patch_embed.proj2):
            nn.init.normal_(projection.weight, mean=0.0, std=self.megatron_init_std)
            if projection.bias is not None:
                nn.init.constant_(projection.bias, 0)

    def _initialize_megatron_output_projections(self, root):
        """
        Initialize attention and FFN output projections with depth-scaled std.
        """
        for module in root.modules():
            if isinstance(module, FLUXAttention):
                output_projection = module.to_out[0]
                nn.init.normal_(output_projection.weight, mean=0.0, std=self.megatron_output_init_std)
            elif isinstance(module, SwiGLUFFN):
                nn.init.normal_(module.w3.weight, mean=0.0, std=self.megatron_output_init_std)
            elif isinstance(module, DeepSeekSharedExpert):
                nn.init.normal_(module.w2.weight, mean=0.0, std=self.megatron_output_init_std)

    def get_hierarchical_pos_embed(self, num_patches, device=None, dtype=None):
        """
        Get cached sin-cos position embeddings for a square patch grid.

        Args
            num_patches (int):
                Number of image patch tokens.
            device (torch.device, optional):
                Device for the returned embeddings.
            dtype (torch.dtype, optional):
                Dtype for the returned embeddings.
        Returns
            torch.Tensor: position embeddings [1, num_patches, hidden_size].
        """
        key = str(num_patches)
        if key not in self.hierarchical_pos_embed_cache:
            grid_size = int(num_patches ** 0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Expected square patch grid, got {num_patches} patches")

            pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
            self.hierarchical_pos_embed_cache[key] = torch.from_numpy(pos_embed).float().unsqueeze(0)

        return self.hierarchical_pos_embed_cache[key].to(device=device, dtype=dtype)

    @torch.no_grad()
    def _prepare_image_ids(self, height, width, device):
        """
        Build height/width coordinates for image-token RoPE.

        Args
            height (int):
                Patch grid height.
            width (int):
                Patch grid width.
            device (torch.device):
                Device for the coordinates.
        Returns
            coordinates [1, height * width, 2], in row-major order.
        """
        height_pos = torch.arange(height, device=device, dtype=torch.float32)
        width_pos = torch.arange(width, device=device, dtype=torch.float32)
        image_ids = torch.stack(torch.meshgrid(height_pos, width_pos, indexing="ij"), dim=-1)

        return rearrange(image_ids.unsqueeze(0), "b h w c -> b (h w) c")


    def _ddt_condition(self, body_tokens, t_emb, hidden_length):
        """
        Combine body features and timestep embeddings to condition the DDT head.

        Args
            body_tokens (torch.Tensor):
                Packed body features [1, total_tokens, hidden_size].
            t_emb (torch.Tensor):
                Timestep embeddings [num_images, hidden_size].
            hidden_length (List[int]):
                Image token count per image, excluding context tokens.
        Returns
            torch.Tensor: DDT conditioning [1, total_tokens, ddt_head_dim].
        """
        t_tokens = expand_segment_embeddings(t_emb, hidden_length, body_tokens.shape[0])
        condition = F.silu(body_tokens + t_tokens)
        return self.ddt_s_projector(condition)

    def forward(self, sample, timesteps, labels):
        """
        Predict stage endpoints conditioned on global timesteps t.

        Args
            sample (torch.Tensor or List[torch.Tensor]):
                Images [stage_batch_size, C, H, W] for each active stage.
            timesteps (torch.Tensor or List[torch.Tensor]):
                Global timesteps [stage_batch_size] for each active stage.
            labels (torch.Tensor):
                Class labels [num_images], concatenated in stage order.
        Returns
            List[torch.Tensor]: endpoint predictions [stage_batch_size, C, H, W] per active stage.
        """
        samples = sample if isinstance(sample, list) else [sample]
        timestep_list = timesteps if isinstance(timesteps, list) else [timesteps]
        if len(samples) != len(timestep_list):
            raise ValueError("sample and timesteps must have the same number of active stages")

        stage_tokens = []
        stage_ddt_tokens = []
        stage_batch_sizes = []
        stage_token_counts = []
        hidden_length = []
        output_shapes = []
        image_rotary_emb = []
        concat_image_rotary_emb = []
        ddt_image_rotary_emb = []

        for stage_sample in samples:
            batch_size, _, image_height, image_width = stage_sample.shape  # [B_stage, C, H, W]
            token_height = image_height // self.patch_size  # Patch grid height.
            token_width = image_width // self.patch_size  # Patch grid width.
            # Embed image patches separately for the body and DDT head.
            tokens = self.x_embedder(stage_sample)  # [B_stage, num_patches, hidden_size]
            tokens = tokens + self.get_hierarchical_pos_embed(tokens.shape[1], tokens.device, tokens.dtype)  # Add [1, num_patches, hidden_size].
            ddt_tokens = self.ddt_x_embedder(stage_sample)  # [B_stage, num_patches, ddt_head_dim]
            token_count = tokens.shape[1]  # num_patches = token_height * token_width

            # Image tokens use grid coordinates; class context tokens use (0, 0).
            base_ids = self._prepare_image_ids(token_height, token_width, tokens.device)  # [1, num_patches, 2]
            label_ids = torch.zeros(
                1,
                self.in_context_len,
                base_ids.shape[-1],
                device=tokens.device,
                dtype=base_ids.dtype,
            )  # [1, in_context_len, 2]
            context_ids = torch.cat([label_ids, base_ids], dim=1)  # [1, in_context_len + num_patches, 2]

            # Pack all images into batch_size=1; hidden_length retains image boundaries.
            stage_tokens.append(rearrange(tokens, "b s d -> 1 (b s) d"))
            stage_ddt_tokens.append(rearrange(ddt_tokens, "b s d -> 1 (b s) d"))
            stage_batch_sizes.append(batch_size)
            stage_token_counts.append(token_count)
            hidden_length.extend([token_count] * batch_size)
            output_shapes.append((token_height, token_width))
            # Images in the same stage share the same positional grid.
            image_rotary_emb.extend([self.hierarchical_rope_embed(base_ids)] * batch_size)
            concat_image_rotary_emb.extend([self.hierarchical_rope_embed(context_ids)] * batch_size)
            ddt_image_rotary_emb.extend([self.ddt_hierarchical_rope_embed(base_ids)] * batch_size)

        # Concatenate stages in the same order for tokens, RoPE, and conditioning.
        image_rotary_emb = torch.cat(image_rotary_emb, dim=1)  # [1, total_tokens, 1, body_head_dim // 2, 2, 2]
        concat_image_rotary_emb = torch.cat(concat_image_rotary_emb, dim=1)  # Image + context tokens, in per-image order.
        ddt_image_rotary_emb = torch.cat(ddt_image_rotary_emb, dim=1)  # [1, total_tokens, 1, ddt_head_dim_per_head // 2, 2, 2]
        hidden_states = torch.cat(stage_tokens, dim=1)  # [1, total_tokens, hidden_size]
        ddt_hidden_states = torch.cat(stage_ddt_tokens, dim=1)  # [1, total_tokens, ddt_head_dim]
        timesteps = torch.cat(timestep_list, dim=0)  # [num_images], in stage order.
        if labels.shape[0] != timesteps.shape[0]:
            raise ValueError("labels must be concatenated in the same stage order as sample")
        t_emb = self.t_embedder(timesteps.reshape(-1))  # [num_images, hidden_size]
        y_emb = self.y_embedder(labels)  # [num_images, hidden_size]
        c = t_emb + y_emb  # [num_images, hidden_size]

        has_label_context = False  # Context tokens are inserted at in_context_start.
        image_hidden_length = list(hidden_length)  # Keep image-only lengths for removing context later.

        def add_label_context(states):
            # Prepend class embeddings plus learned context positions to each image.
            stage_segments = []
            token_offset = 0  # Start of the current stage in packed image tokens.
            label_offset = 0  # Start of the current stage in y_emb.
            next_hidden_length = []  # Token count per image, including context.

            for batch_size, token_count in zip(stage_batch_sizes, stage_token_counts):
                stage_state = states[:, token_offset:token_offset + batch_size * token_count]  # [1, B_stage * token_count, hidden_size]
                stage_state = stage_state.reshape(batch_size, token_count, -1)  # [B_stage, token_count, hidden_size]
                context = y_emb[label_offset:label_offset + batch_size].unsqueeze(1).repeat(1, self.in_context_len, 1)  # [B_stage, in_context_len, hidden_size]
                context = context + self.in_context_posemb  # Add learned positions [1, in_context_len, hidden_size].
                stage_state = torch.cat([context, stage_state], dim=1)  # [B_stage, in_context_len + token_count, hidden_size]
                stage_segments.append(rearrange(stage_state, "b s d -> 1 (b s) d"))  # Pack [context_1, image_1, context_2, image_2, ...].
                next_hidden_length.extend([token_count + self.in_context_len] * batch_size)
                token_offset += batch_size * token_count
                label_offset += batch_size

            return torch.cat(stage_segments, dim=1), next_hidden_length  # [1, total_tokens + num_images * in_context_len, hidden_size], lengths.

        def strip_label_context(states):
            # Remove each image's context prefix and restore image-only token lengths.
            stage_segments = []
            token_offset = 0  # Start of the current stage in packed context + image tokens.

            for batch_size, token_count in zip(stage_batch_sizes, stage_token_counts):
                full_count = token_count + self.in_context_len  # Context + image token count per image.
                stage_state = states[:, token_offset:token_offset + batch_size * full_count]  # [1, B_stage * full_count, hidden_size]
                stage_state = stage_state.reshape(batch_size, full_count, -1)  # [B_stage, full_count, hidden_size]
                stage_state = stage_state[:, self.in_context_len:]  # [B_stage, token_count, hidden_size]
                stage_segments.append(rearrange(stage_state, "b s d -> 1 (b s) d"))  # [1, B_stage * token_count, hidden_size]
                token_offset += batch_size * full_count

            return torch.cat(stage_segments, dim=1), list(image_hidden_length)  # [1, total_tokens, hidden_size], image-only lengths.

        # Insert class context at in_context_start, then process the remaining body blocks.
        for i, block in enumerate(self.blocks):
            if i == self.in_context_start:
                hidden_states, hidden_length = add_label_context(hidden_states)
                has_label_context = True

            hidden_states = block(
                hidden_states,
                c,
                hidden_length=hidden_length,
                image_rotary_emb=concat_image_rotary_emb if has_label_context else image_rotary_emb,
            )

        # The DDT head receives image tokens only.
        hidden_states, hidden_length = strip_label_context(hidden_states)

        ddt_condition = self._ddt_condition(hidden_states, t_emb, hidden_length)
        for block in self.ddt_blocks:
            ddt_hidden_states = block(
                ddt_hidden_states,
                ddt_condition,
                hidden_length=hidden_length,
                image_rotary_emb=ddt_image_rotary_emb,
            )

        pred = self.ddt_final_layer(ddt_hidden_states, ddt_condition)

        # Unpack each stage and reshape predicted patch pixels back to images.
        outputs = []
        token_offset = 0
        for batch_size, token_count, (height, width) in zip(stage_batch_sizes, stage_token_counts, output_shapes):
            stage_pred = pred[:, token_offset:token_offset + batch_size * token_count]
            stage_pred = stage_pred.reshape(
                shape=(batch_size, height, width, self.patch_size, self.patch_size, -1)
            )
            stage_pred = rearrange(stage_pred, "b h w p1 p2 c -> b (h p1) (w p2) c")
            outputs.append(rearrange(stage_pred, "b h w c -> b c h w"))
            token_offset += batch_size * token_count
        return outputs


# Model variants

def PixelMoE_B_E32A4_8(**kwargs):
    return PixelMoE(depth=12, hidden_size=768, num_heads=12,
                    bottleneck_dim=128, in_context_len=32, in_context_start=4,
                    patch_size=8,
                    ddt_head_dim=1536, ddt_head_depth=2, ddt_head_num_heads=12,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=384, moe_shared_inter_dim=632,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


def PixelMoE_B_E32A4_16(**kwargs):
    return PixelMoE(depth=12, hidden_size=768, num_heads=12,
                    bottleneck_dim=128, in_context_len=32, in_context_start=4,
                    patch_size=16,
                    ddt_head_dim=1536, ddt_head_depth=2, ddt_head_num_heads=12,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=384, moe_shared_inter_dim=632,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


def PixelMoE_L_E32A4_8(**kwargs):
    return PixelMoE(depth=24, hidden_size=1024, num_heads=16,
                    bottleneck_dim=128, in_context_len=32, in_context_start=8,
                    patch_size=8,
                    ddt_head_dim=2048, ddt_head_depth=2, ddt_head_num_heads=16,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=512, moe_shared_inter_dim=836,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


def PixelMoE_L_E32A4_16(**kwargs):
    return PixelMoE(depth=24, hidden_size=1024, num_heads=16,
                    bottleneck_dim=128, in_context_len=32, in_context_start=8,
                    patch_size=16,
                    ddt_head_dim=2048, ddt_head_depth=2, ddt_head_num_heads=16,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=512, moe_shared_inter_dim=836,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


def PixelMoE_H_E32A4_8(**kwargs):
    return PixelMoE(depth=32, hidden_size=1280, num_heads=16,
                    bottleneck_dim=256, in_context_len=32, in_context_start=10,
                    patch_size=8,
                    ddt_head_dim=2560, ddt_head_depth=4, ddt_head_num_heads=16,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=640, moe_shared_inter_dim=1048,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


def PixelMoE_H_E32A4_16(**kwargs):
    return PixelMoE(depth=32, hidden_size=1280, num_heads=16,
                    bottleneck_dim=256, in_context_len=32, in_context_start=10,
                    patch_size=16,
                    ddt_head_dim=2560, ddt_head_depth=4, ddt_head_num_heads=16,
                    moe_num_experts=32, moe_top_k=4,
                    moe_routed_inter_dim=640, moe_shared_inter_dim=1048,
                    moe_route_scale=2.0, moe_n_dense_layers=4, **kwargs)


PixelMoE_models = {
    "PixelMoE-B-E32A4/8": PixelMoE_B_E32A4_8,
    "PixelMoE-B-E32A4/16": PixelMoE_B_E32A4_16,
    "PixelMoE-L-E32A4/8": PixelMoE_L_E32A4_8,
    "PixelMoE-L-E32A4/16": PixelMoE_L_E32A4_16,
    "PixelMoE-H-E32A4/8": PixelMoE_H_E32A4_8,
    "PixelMoE-H-E32A4/16": PixelMoE_H_E32A4_16,
}
