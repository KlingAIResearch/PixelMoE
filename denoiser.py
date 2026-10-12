import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_pixelmoe import PixelMoE_models
from hierarchical_image_noise import HierarchicalImageNoiseBuilder


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = PixelMoE_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            proj_drop=args.proj_dropout,
            flux_attn_proj_dropout=args.flux_attn_proj_dropout,
            use_flash_attn=args.use_flash_attn,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.sigma_min = args.sigma_min
        self.noise_scale = args.noise_scale

        # Hierarchical Flow training hyperparameters.
        self.hierarchical_stages = list(args.hierarchical_stages)
        self.hierarchical_logit_mean = args.hierarchical_logit_mean
        self.hierarchical_logit_std = args.hierarchical_logit_std
        self.hierarchical_noise_rho = args.hierarchical_noise_rho
        self.hierarchical_min_stage_sigma = float(args.hierarchical_min_stage_sigma)
        self.hierarchical_noise_builder = HierarchicalImageNoiseBuilder(
            stages=self.hierarchical_stages,
            end_timesteps=args.hierarchical_end_timesteps,
            noise_rho=self.hierarchical_noise_rho,
            noise_scale=self.noise_scale,
            logit_mean=self.hierarchical_logit_mean,
            logit_std=self.hierarchical_logit_std,
            stage_logit_means=args.hierarchical_stage_logit_means,
            stage_logit_stds=args.hierarchical_stage_logit_stds,
            min_stage_sigma=self.hierarchical_min_stage_sigma,
        )

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.use_second_ema = args.use_second_ema
        self.ema_params1 = None
        self.ema_params2 = None
        self.ema_buffer_names = []
        self.ema_buffers1 = None
        self.ema_buffers2 = None

        # generation hyper-parameters
        self.method = args.sampling_method
        self.sampling_steps = list(args.num_sampling_steps)
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels
        return self.forward_hierarchical(x, labels_dropped)

    def forward_hierarchical(self, x, labels):
        # Assign images to stages and prepare noisy inputs, timesteps, and endpoints.
        hierarchical_noise = self.hierarchical_noise_builder(x)
        # Select labels using the original batch indices of each stage's images.
        stage_labels_list = [
            labels.index_select(0, sample_indices.to(labels.device))
            for sample_indices in hierarchical_noise.sample_indices_list
        ]
        # Predict one endpoint per image; labels follow the same stage order as inputs.
        pred_segments = self.net(
            hierarchical_noise.noisy_images_list,
            hierarchical_noise.timesteps_list,
            torch.cat(stage_labels_list, dim=0),
        )

        losses = []
        # endpoint MSE / max(sigma, sigma_min)^2.
        for pred, sigmas, endpoint in zip(
            pred_segments,
            hierarchical_noise.sigmas_list,
            hierarchical_noise.targets_list,
        ):
            sigma_safe = sigmas.float().clamp_min(self.sigma_min)
            loss_error = (pred.float() - endpoint.float()) / sigma_safe
            # Average over channels and pixels, keeping one loss value per image.
            loss = torch.mean(loss_error.pow(2).reshape(endpoint.shape[0], -1), dim=1)
            losses.append(loss)

        # Average over all images, so stages contribute according to their image counts.
        loss = torch.cat(losses, dim=0).mean()
        return loss

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def generate(self, labels):
        return self.generate_hierarchical(labels)

    @torch.no_grad()
    def generate_hierarchical(self, labels):
        """Generate images from low to high resolution.

        Each higher-resolution stage starts from the previous output,
        upsampled and re-noised to match its starting distribution.
        """
        device = labels.device
        dtype = next(self.net.parameters()).dtype
        batch_size = labels.size(0)
        num_stages = len(self.hierarchical_stages)

        # Downsample full-resolution noise to the first stage. Multiplying by 2
        # restores the noise variance after each 2x bilinear downsampling.
        z = self.noise_scale * torch.randn(
            batch_size,
            3,
            self.img_size,
            self.img_size,
            device=device,
            dtype=dtype,
        )
        height = self.img_size
        width = self.img_size
        for _ in range(num_stages - 1):
            height //= 2
            width //= 2
            z = F.interpolate(z, size=(height, width), mode="bilinear", align_corners=False) * 2

        for stage_index in range(num_stages):
            if stage_index > 0:
                height *= 2
                width *= 2
                z = F.interpolate(z, size=(height, width), mode="nearest")
                z = self._hierarchical_transition_renoise(z, stage_index)

            z = self._sample_hierarchical_stage(
                z,
                stage_index,
                labels,
            )

        return z

    def _hierarchical_transition_renoise(self, latents, stage_index):
        """Rescale the upsampled image and add noise for the next stage."""
        previous_t_end = self.hierarchical_noise_builder.end_timesteps[stage_index - 1]
        t_start = self.hierarchical_noise_builder.start_timesteps[stage_index]

        rho = self.hierarchical_noise_rho
        alpha = t_start / math.sqrt(1. - rho)
        signal_scale = (1. - t_start) / (1. - previous_t_end)

        # Rescale the previous output and add block-correlated noise with the
        # same noise_scale used during training.
        noise = self.sample_block_noise_like(latents) * self.noise_scale
        return signal_scale * latents + alpha * noise

    def sample_block_noise_like(self, latents):
        """Sample correlated noise within each spatial 2x2 block.

        Each block follows a 4D Gaussian with covariance:
            cov = (1 - rho) * I + rho * 11^T

        For rho=-1/3, the four pixels in a block are negatively correlated and
        their sum has zero variance.
        """
        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Hierarchical block noise requires even spatial sizes")

        rho = self.hierarchical_noise_rho
        covariance = (
            torch.eye(4, device=latents.device, dtype=torch.float32) * (1 - rho)
            + torch.ones(4, 4, device=latents.device, dtype=torch.float32) * rho
        )
        mean = torch.zeros(4, device=latents.device, dtype=torch.float32)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        block_number = batch_size * channels * (height // 2) * (width // 2)
        noise = dist.sample((block_number,)).to(dtype=latents.dtype)
        noise = noise.reshape(batch_size, channels, height // 2, width // 2, 2, 2)
        noise = noise.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, height, width)
        return noise

    @torch.no_grad()
    def _sample_hierarchical_stage(self, z, stage_index, labels):
        """Denoise z through one stage using Euler or Heun steps."""
        global_timesteps, sigmas = self._hierarchical_stage_timesteps(stage_index, z.device, z.dtype)

        for i in range(len(sigmas) - 1):
            t = global_timesteps[i].expand(z.shape[0]).to(device=z.device, dtype=z.dtype)
            sigma = sigmas[i].expand(z.shape[0]).view(-1, 1, 1, 1)
            sigma_next = sigmas[i + 1].expand(z.shape[0]).view(-1, 1, 1, 1)

            if self.method == "euler":
                z = self._hierarchical_euler_step(
                    z, t, sigma, sigma_next, labels
                )
            elif self.method == "heun":
                t_next = global_timesteps[i + 1].expand(z.shape[0]).to(device=z.device, dtype=z.dtype)
                z = self._hierarchical_heun_step(
                    z, t, t_next, sigma, sigma_next, labels
                )
            else:
                raise NotImplementedError

        return z

    def _hierarchical_stage_timesteps(self, stage_index, device, dtype):
        """Select this stage's grid points and return global timestep and stage-relative sigma."""
        t_start = self.hierarchical_noise_builder.start_timesteps[stage_index]
        t_end = self.hierarchical_noise_builder.end_timesteps[stage_index]
        # A zero lower bound covers the full stage interval. A positive value
        # excludes near-endpoint grid points, except for the endpoint added below.
        min_sigma = self.hierarchical_min_stage_sigma
        # Convert the stage-relative lower bound to global t. The true endpoint
        # t_end is added below so sampling still finishes at sigma=0.
        sampling_t_end = t_end + min_sigma * (t_start - t_end)
        sampling_t_start = t_start
        global_timesteps = self._hierarchical_global_sampling_timesteps(device, dtype)
        global_timesteps = global_timesteps[
            (global_timesteps <= sampling_t_start) & (global_timesteps >= sampling_t_end)
        ]
        global_timesteps = torch.cat([
            global_timesteps,
            torch.tensor([sampling_t_start, sampling_t_end, t_end], device=device, dtype=dtype),
        ])
        global_timesteps = torch.unique(global_timesteps, sorted=True).flip(0)
        global_timesteps[0] = sampling_t_start
        global_timesteps[-1] = t_end

        denom = max(t_start - t_end, torch.finfo(dtype).eps)
        sigmas = (global_timesteps - t_end) / denom
        sigmas[-1] = 0.0
        return global_timesteps, sigmas

    def _hierarchical_global_sampling_timesteps(self, device, dtype):
        """Build the shared global timestep grid from t=1 to t=0."""
        if len(self.sampling_steps) == 1:
            total_steps = self.sampling_steps[0] * len(self.hierarchical_stages)
        else:
            total_steps = sum(self.sampling_steps)

        timesteps = torch.linspace(1.0, 0.0, total_steps + 1, device=device, dtype=dtype)
        timesteps = self._logit_shift_sampling_timesteps(timesteps)
        return timesteps

    def _logit_shift_sampling_timesteps(self, timesteps):
        """Adjust timestep spacing using the global Logit-Normal mean and std."""
        eps = torch.finfo(timesteps.dtype).eps
        t = timesteps.clamp(eps, 1.0 - eps)
        exp_mean = math.exp(float(self.hierarchical_logit_mean))
        shifted = exp_mean / (
            exp_mean + torch.pow(1.0 / t - 1.0, float(self.hierarchical_logit_std))
        )
        shifted[0] = 1.0
        shifted[-1] = 0.0
        return shifted

    @torch.no_grad()
    def _hierarchical_euler_step(self, z, t, sigma, sigma_next, labels):
        """Euler step: take one Euler step using the current velocity and change in sigma."""
        velocity = self._forward_hierarchical_sample(
            z,
            t,
            sigma,
            labels,
        )
        return z + (sigma_next - sigma) * velocity

    @torch.no_grad()
    def _hierarchical_heun_step(self, z, t, t_next, sigma, sigma_next, labels):
        """Heun step: average current and next velocities for a Heun step; use Euler at sigma=0."""
        velocity = self._forward_hierarchical_sample(
            z,
            t,
            sigma,
            labels,
        )
        z_euler = z + (sigma_next - sigma) * velocity
        if torch.all(sigma_next <= 0):
            return z_euler

        velocity_next = self._forward_hierarchical_sample(
            z_euler,
            t_next,
            sigma_next,
            labels,
        )
        return z + (sigma_next - sigma) * 0.5 * (velocity + velocity_next)

    @torch.no_grad()
    def _forward_hierarchical_sample(self, z, t, sigma, labels):
        """Predict endpoint-derived velocity, applying CFG within its global-timestep interval."""
        pred_cond = self.net(z, t, labels)[0]
        v_cond = self._hierarchical_prediction_to_velocity(
            pred_cond,
            z,
            sigma,
        )

        if self.cfg_scale == 1.0 or self.label_drop_prob <= 0.0:
            return v_cond

        uncond_labels = torch.full(
            labels.shape,
            self.num_classes,
            device=labels.device,
            dtype=labels.dtype,
        )
        pred_uncond = self.net(z, t, uncond_labels)[0]
        v_uncond = self._hierarchical_prediction_to_velocity(
            pred_uncond,
            z,
            sigma,
        )

        cfg_t = t.view(-1, 1, 1, 1)
        low, high = self.cfg_interval
        lower_ok = (low == 0) | (cfg_t > low)
        upper_ok = (high == 1) | (cfg_t < high)
        interval_mask = lower_ok & upper_ok
        cfg_scale = torch.where(interval_mask, self.cfg_scale, 1.0)

        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_guided

    def _hierarchical_prediction_to_velocity(self, pred, noisy, sigma):
        """Convert an endpoint prediction to velocity with a clamped sigma denominator."""
        # v_theta = (x_sigma - predicted endpoint) / sigma.
        dtype_eps = torch.finfo(sigma.dtype).eps
        sampling_eps = float(self.sigma_min)
        sigma_eps = max(sampling_eps, dtype_eps)
        return (noisy - pred) / sigma.clamp_min(sigma_eps)

    @torch.no_grad()
    def update_moe_router_bias(self):
        """Update each MoE router's balancing bias using this batch's expert loads."""
        for module in self.net.modules():
            update_router_bias = getattr(module, "update_router_bias_from_latest", None)
            if update_router_bias is not None:
                update_router_bias()

    @torch.no_grad()
    def clear_moe_router_load_counts(self):
        """Discard this batch's routing counts when its optimizer update is skipped."""
        for module in self.net.modules():
            if hasattr(module, "load_counts"):
                module.load_counts = None

    def ema_buffer_items(self):
        """Return the names and tensors of router bias buffers tracked by EMA."""
        return [
            (name, buffer)
            for name, buffer in self.named_buffers()
            if name.endswith("router_bias")
        ]

    @torch.no_grad()
    def init_ema_buffers(self, ema_state_dict1=None, ema_state_dict2=None):
        """Initialize router bias EMA tensors from checkpoint values or current buffers."""
        items = self.ema_buffer_items()
        self.ema_buffer_names = [name for name, _ in items]
        if not items:
            self.ema_buffers1 = []
            self.ema_buffers2 = [] if self.use_second_ema else None
            return
        self.ema_buffers1 = [
            (ema_state_dict1[name] if ema_state_dict1 is not None else buffer)
            .detach()
            .to(device=buffer.device, dtype=buffer.dtype)
            .clone()
            for name, buffer in items
        ]
        if self.use_second_ema:
            self.ema_buffers2 = [
                (ema_state_dict2[name] if ema_state_dict2 is not None else buffer)
                .detach()
                .to(device=buffer.device, dtype=buffer.dtype)
                .clone()
                for name, buffer in items
            ]
        else:
            self.ema_buffers2 = None

    @torch.no_grad()
    def update_ema(self):
        """Update EMA copies of model parameters and router bias buffers."""
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        if self.use_second_ema and self.ema_params2 is not None:
            for targ, src in zip(self.ema_params2, source_params):
                targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
        if self.ema_buffers1:
            source_buffers = dict(self.named_buffers())
            for name, targ in zip(self.ema_buffer_names, self.ema_buffers1):
                src = source_buffers[name]
                targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
            if self.use_second_ema and self.ema_buffers2:
                for name, targ in zip(self.ema_buffer_names, self.ema_buffers2):
                    src = source_buffers[name]
                    targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

    def apply_ema_buffers_to_state_dict(self, state_dict, ema_index=1):
        """Write the selected EMA router bias values into the state dict for saving."""
        buffers = self.ema_buffers1 if ema_index == 1 else self.ema_buffers2
        if buffers is None:
            return state_dict
        for name, value in zip(self.ema_buffer_names, buffers):
            state_dict[name] = value
        return state_dict
