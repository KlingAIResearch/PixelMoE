import torch
from torch.optim import Optimizer


def _is_stacked_moe_expert_parameter(name, param):
    """Return whether ``param`` is a stacked DeepSeek expert matrix.

    DeepSeek keeps each expert projection as one ``[experts, rows, cols]``
    parameter.  gram-newton-schulz.Muon natively splits the leading expert
    dimension and orthogonalizes every ``[rows, cols]`` slice independently.
    Keep the name check narrow so unrelated trainable 3D tensors (for example
    learned positional embeddings) remain on AdamW.
    """
    return param.ndim == 3 and name.endswith((
        "mlp.experts.w1",
        "mlp.experts.w2",
        "mlp.experts.w3",
    ))


class GMuonWithAdamW(Optimizer):
    """GMuon + AdamW composite optimizer.

    Every 2D parameter goes to ``gram_newton_schulz.Muon`` with
    ``adjust_lr="rms_norm"``, except named
    exclusions in ``skip_keywords``. These and all other parameters go to AdamW.
    Stacked 3D DeepSeek expert matrices
    also go to GMuon, which treats the
    leading expert dimension as a batch of independent 2D matrices.
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        muon_momentum=0.95,
        muon_nesterov=True,
        ns_coefficients_preset="POLAR_EXPRESS_COEFFICIENTS",
        adjust_lr="rms_norm",
        skip_keywords=(),
    ):
        try:
            from gram_newton_schulz import Muon as GMuon
        except ImportError as exc:
            raise ImportError(
                "GMuon requires gram-newton-schulz. Install it with:\n"
                '  pip install "gram-newton-schulz @ git+https://github.com/nanovisionx/gmuon.git@main"'
            ) from exc

        params = []
        for name, param in parameters:
            if not param.requires_grad:
                continue
            # Main excludes AdaLN, prediction heads, and timestep/class embeddings.
            keyword_skip = any(keyword and keyword in name for keyword in skip_keywords)
            use_muon = (
                (param.ndim == 2 or _is_stacked_moe_expert_parameter(name, param))
                and not keyword_skip
            )
            params.append((use_muon, param))

        muon_params = [p for use_muon, p in params if use_muon]
        adamw_params = [p for use_muon, p in params if not use_muon]
        if not muon_params:
            raise ValueError("No 2D parameters were assigned to GMuon")

        self._muon = GMuon(
            muon_params,
            lr=lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            weight_decay=weight_decay,
            ns_coefficients_preset=ns_coefficients_preset,
            ns_use_kernels=False,
            adjust_lr=adjust_lr,
        )
        self._adamw = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=1e-8,
        )
        # Share the underlying group objects so the LR scheduler updates both optimizers.
        self.param_groups = self._muon.param_groups + self._adamw.param_groups
        self.defaults = {}

    @property
    def state(self):
        merged = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none=False):
        # The parameter sets are disjoint; clear gradients in both optimizers.
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # Apply each algorithm once to its own parameters after a single backward pass.
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self):
        # Preserve separate momentum/Adam states in the checkpoint.
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups
