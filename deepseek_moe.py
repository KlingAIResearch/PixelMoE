"""DeepSeek-style token-choice MoE using PyTorch grouped matrix multiply.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def normalize_router_bias_update(imbalance, eps=1e-6):
    """RMS-normalize the expert-load imbalance vector."""
    return imbalance * torch.rsqrt(imbalance.square().mean() + float(eps))


class DeepSeekMoEGate(nn.Module):
    """Select top-k experts using FP32 sigmoid scores and balancing bias."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        route_scale: float = 1.0,
        router_bias_update_rate: float = 1e-3,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("moe_num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")

        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = float(route_scale)
        self.router_bias_update_rate = float(router_bias_update_rate)

        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        # Balancing bias is checkpointed and updated without gradients.
        self.register_buffer(
            "router_bias", 
            torch.zeros(num_experts, dtype=torch.float32), 
            persistent=True
        )
        # Cache this batch's expert loads until the optimizer update is accepted.
        self.load_counts = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.router_bias.zero_()

    def forward(self, hidden_states):
        # Route packed tokens [num_tokens, dim] in FP32.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            logits = F.linear(
                hidden_states.type(torch.float32),
                self.weight.type(torch.float32),
            )

        scores = logits.sigmoid()

        # Bias affects expert selection; aggregating weights use the unbiased scores.
        selection_scores = scores + self.router_bias.to(
            device=scores.device, dtype=scores.dtype
        ).detach()
        indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        # Normalize each token's selected weights, then apply the routing scale.
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.route_scale
        if self.training:
            self.record_load_counts(indices)
        return weights, indices

    @torch.no_grad()
    def record_load_counts(self, topk_idx):
        # Each selected token contributes one count to each of its top-k experts.
        counts = topk_idx.new_zeros(
            (topk_idx.shape[0], self.num_experts),
            dtype=torch.float32,
        )
        counts.scatter_(1, topk_idx, 1)
        self.load_counts = counts.sum(dim=0)

    @torch.no_grad()
    def update_bias(self, update_rate=1e-3):
        # Called after optimizer.step(); skipped batches clear their counts instead.
        if self.load_counts is None:
            return
        counts = self.load_counts
        if dist.is_initialized():
            # Use the same averaged expert loads on every DDP rank.
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            counts /= dist.get_world_size()
        # RMS-normalize the deficit: underloaded experts gain bias, overloaded lose it.
        delta = normalize_router_bias_update(counts.mean() - counts)
        self.router_bias.add_(
            delta.to(dtype=self.router_bias.dtype),
            alpha=float(update_rate),
        )
        self.load_counts = None

class DeepSeekTokenReorderer(nn.Module):
    """Stably group token-expert assignments by expert index."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, top_scores, selected_experts_indices, token_ids):
        """Group N * top_k token-expert assignments by expert.

        Returns:
            top_scores_experts_sorted: [N * top_k] mixing weights in expert
                order, used to weight expert outputs.
            token_indices_experts_sorted: [N * top_k] original token IDs in
                the same order, used to gather inputs and scatter outputs.
                Each token appears top_k times.
            num_tokens_per_expert: [num_experts] route counts defining each
                expert's contiguous input segment.
        """
        flat_experts = selected_experts_indices.reshape(-1)
        if flat_experts.is_cuda:
            num_tokens_per_expert = torch.histc(
                flat_experts,
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )
        else:
            # CPU histc does not support the int64 indices returned by topk.
            num_tokens_per_expert = torch.bincount(
                flat_experts,
                minlength=self.num_experts,
            )
        sorted_idx = torch.argsort(flat_experts, stable=True)
        top_scores_experts_sorted = top_scores.reshape(-1).index_select(0, sorted_idx)
        token_indices_experts_sorted = token_ids.reshape(-1).index_select(0, sorted_idx)
        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run packed SwiGLU experts with grouped matrix multiplication."""
    # Input rows are already grouped by expert; each row is one token-expert route.
    if x.dim() != 2:
        raise ValueError(f"grouped expert input must be 2D, got shape {tuple(x.shape)}")
    if num_tokens_per_expert.numel() != w1.shape[0]:
        raise ValueError("num_tokens_per_expert must contain one count per expert")

    # Skip experts with no assigned tokens, preserving the order of active experts.
    non_zero_mask = num_tokens_per_expert > 0
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return x.new_empty((0, w2.shape[1]))
    if len(non_zero_indices) < len(num_tokens_per_expert):
        counts = num_tokens_per_expert[non_zero_mask]
        active_w1 = w1[non_zero_mask]
        active_w2 = w2[non_zero_mask]
        active_w3 = w3[non_zero_mask]
    else:
        # Every expert has tokens, so use all expert weights directly.
        counts = num_tokens_per_expert
        active_w1 = w1
        active_w2 = w2
        active_w3 = w3
    # Cumulative segment ends.
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

    # Compute each expert's gate (w1) and value (w3) projections in BF16.
    x_bf16 = x.bfloat16()
    gate = torch._grouped_mm(
        x_bf16,
        active_w1.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    linear = torch._grouped_mm(
        x_bf16,
        active_w3.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    # SwiGLU: multiply the SiLU-activated gate by the value projection.
    activated_gate = F.silu(gate)
    hidden = activated_gate * linear
    # Project through w2, keeping the same expert-grouped row order.
    output = torch._grouped_mm(
        hidden,
        active_w2.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    return output.type_as(x)


class DeepSeekGroupedExperts(nn.Module):
    """Routed experts with matrices stacked along the expert dimension."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        num_layers: int,
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, output_dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.num_layers = num_layers
        self.reset_parameters()

    def forward(self, x, num_tokens_per_expert):
        return _run_experts_grouped_mm(
            self.w1,
            self.w2,
            self.w3,
            x,
            num_tokens_per_expert,
        )

    def reset_parameters(self):
        # Initialize expert projections with a truncated normal distribution.
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=0.02)
        # Scale the residual output projection by the transformer body depth.
        self.w2.data /= (2 * self.num_layers) ** 0.5


class DeepSeekSharedExpert(nn.Module):
    """Dense SwiGLU shared expert used for every token."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # Shared SwiGLU processes every image token independently of routed selection.
        gate = self.w1(x)
        linear = self.w3(x)
        activated_gate = F.silu(gate)
        product = activated_gate * linear
        output = self.w2(product)
        return output


class DeepSeekMoEFFN(nn.Module):
    """Standalone DeepSeek-style routed and shared expert FFN."""

    def __init__(
        self,
        dim: int,
        routed_inter_dim: int,
        shared_inter_dim: int,
        num_layers: int,
        num_experts: int = 8,
        top_k: int = 2,
        route_scale: float = 1.0,
        router_bias_update_rate: float = 1e-3,
        drop: float = 0.0,
    ):
        super().__init__()
        if routed_inter_dim is None or shared_inter_dim is None:
            raise ValueError("MoE requires explicit routed_inter_dim and shared_inter_dim")
        expert_hidden_dim = int(routed_inter_dim)
        shared_hidden_dim = int(shared_inter_dim)
        self.num_experts = int(num_experts)
        self.top_k = top_k
        self.gate = DeepSeekMoEGate(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            route_scale=route_scale,
            router_bias_update_rate=router_bias_update_rate,
        )
        self.experts = DeepSeekGroupedExperts(
            dim=dim,
            hidden_dim=expert_hidden_dim,
            output_dim=dim,
            num_experts=self.num_experts,
            num_layers=num_layers,
        )
        self.reorderer = DeepSeekTokenReorderer(num_experts=self.num_experts)
        self.shared_experts = DeepSeekSharedExpert(dim, shared_hidden_dim)
        self.ffn_dropout = nn.Dropout(drop)

    @torch.no_grad()
    def update_router_bias_from_latest(self):
        self.gate.update_bias(self.gate.router_bias_update_rate)

    @torch._dynamo.disable
    def forward(self, hidden_states):
        # Merge packed token dimensions, then select top-k experts per token.
        original_shape = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, original_shape[-1])
        topk_weight, topk_idx = self.gate(hidden_states_flat)

        # A token appears once per selected expert; token_ids allow restoring the original order.
        token_ids = torch.arange(
            hidden_states_flat.shape[0],
            device=hidden_states.device,
        ).unsqueeze(1).expand(-1, self.top_k)
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(topk_weight, topk_idx, token_ids.reshape(-1))

        # Repeat each token ID across its features to gather complete rows.
        gather_index = token_indices_experts_sorted.reshape(-1, 1).expand(
            -1,
            hidden_states_flat.shape[-1],
        )
        routed_input = torch.gather(hidden_states_flat, dim=0, index=gather_index)
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        # Weight routed outputs in FP32, then cast to the input dtype.
        routed_output = (
            routed_output.float()
            * top_scores_experts_sorted.float().unsqueeze(-1)
        ).to(dtype=hidden_states.dtype)

        # Start with the shared expert, then add weighted routed outputs back to each token.
        output = self.shared_experts(hidden_states_flat)
        if self.training or torch.are_deterministic_algorithms_enabled():
            output = output.scatter_add(dim=0, index=gather_index, src=routed_output)
        else:
            # Use the native deterministic reduction in inference.
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                output = output.scatter_add(dim=0, index=gather_index, src=routed_output)
            finally:
                torch.use_deterministic_algorithms(False, warn_only=warn_only)
        output = self.ffn_dropout(output)
        return output.reshape(original_shape)
