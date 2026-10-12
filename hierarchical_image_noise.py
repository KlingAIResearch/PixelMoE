# --------------------------------------------------------
# References:
# Pyramid Flow: https://github.com/jy0205/Pyramid-Flow
# --------------------------------------------------------

import math
from typing import List, NamedTuple

import torch
import torch.nn.functional as F


class HierarchicalImageNoiseOutput(NamedTuple):
    # One entry per active stage in this batch.
    noisy_images_list: List[torch.Tensor]
    # Stage-relative timestep sigma; shape [stage_batch_size, 1, 1, 1].
    sigmas_list: List[torch.Tensor]
    # Global timestep t used to condition the model; shape [stage_batch_size].
    timesteps_list: List[torch.Tensor]
    # Stage endpoints; intermediate-stage endpoints can contain noise.
    targets_list: List[torch.Tensor]
    sample_indices_list: List[torch.Tensor]


class HierarchicalImageNoiseBuilder:
    """
    Prepare noisy images and prediction targets for each resolution stage.

    The input is a batch of images with shape [B, C, H, W]. Each image is
    assigned to one stage for this training step; the number of images per
    stage is determined by the fixed-density allocation. Sample a noise level
    within the assigned stage using that stage's Logit-Normal distribution.

    t is the global timestep and sigma is the timestep
    within a stage: t = t_end + sigma * (t_start - t_end). Construct the noisy
    image as x_sigma = sigma * x_start + (1 - sigma) * x_end.

    Return noisy images, stage-relative sigmas, global timesteps, endpoint
    targets, and original batch indices grouped by stage. Image tensors use
    [stage_batch_size, C, stage_H, stage_W].
    """

    def __init__(
        self,
        stages=(0, 1),
        end_timesteps=(0.75, 0.0),
        noise_rho=-1.0 / 3.0,
        noise_scale=1.0,
        logit_mean=0.0,
        logit_std=1.0,
        stage_logit_means=None,
        stage_logit_stds=None,
        min_stage_sigma=0.0,
    ):
        self.stages = list(stages)
        self.noise_scale = float(noise_scale)
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.stage_logit_means = [float(value) for value in (
            [0.6] * len(self.stages) if stage_logit_means is None else stage_logit_means
        )]
        self.stage_logit_stds = [float(value) for value in (
            [0.8] * len(self.stages) if stage_logit_stds is None else stage_logit_stds
        )]
        self.min_stage_sigma = float(min_stage_sigma)

        if not 0.0 <= self.min_stage_sigma < 1.0:
            raise ValueError("min_stage_sigma must be in [0, 1)")
        if len(self.stage_logit_means) != len(self.stages):
            raise ValueError("stage_logit_means must contain one value per hierarchical stage")
        if len(self.stage_logit_stds) != len(self.stages):
            raise ValueError("stage_logit_stds must contain one value per hierarchical stage")
        if any(std <= 0.0 for std in self.stage_logit_stds):
            raise ValueError("stage_logit_stds values must be positive")
        self.end_timesteps = self._parse_end_timesteps(end_timesteps)
        self.start_timesteps = self._compute_start_timesteps(self.end_timesteps, noise_rho)
        self.training_end_timesteps = [
            t_end + self.min_stage_sigma * (t_start - t_end)
            for t_start, t_end in zip(self.start_timesteps, self.end_timesteps)
        ]
        self.fixed_density_bins = self._build_fixed_density_bins()

    @torch.no_grad()
    def __call__(self, x):
        """
        Build resolution stages and prepare their noisy training inputs.

        Args
            x (torch.Tensor):
                Clean full-resolution images [B, C, H, W].
        Returns
            HierarchicalImageNoiseOutput: training tensors grouped by active stage.
        """
        images_list = self.get_hierarchical_images(x)
        return self.add_continuous_hierarchical_noise(images_list)

    # Stage intervals

    def _parse_end_timesteps(self, end_timesteps):
        # Convert stage endpoints to floats; each stage needs one end timestep.
        values = [float(v) for v in end_timesteps]
        if len(values) == len(self.stages):
            end_timesteps = values
        else:
            raise ValueError("end_timesteps must contain one end t per hierarchical stage")

        # Global timesteps lie in [0, 1].
        if any(s < 0.0 or s > 1.0 for s in end_timesteps):
            raise ValueError("end timesteps must be in [0, 1]")
        # Later stages must end at an equal or smaller timestep, e.g. [0.75, 0.0].
        if any(end_timesteps[i] < end_timesteps[i + 1] for i in range(len(end_timesteps) - 1)):
            raise ValueError("end timesteps must be monotonically non-increasing")
        return end_timesteps

    @staticmethod
    def _compute_start_timesteps(end_timesteps, rho):
        start_timesteps = []
        for i_s, _ in enumerate(end_timesteps):
            # The first stage starts from pure noise at t=1.
            if i_s == 0:
                start_timesteps.append(1.0)
                continue
            # Match the next stage's start distribution after upsampling, rescaling, and re-noising.
            previous_t_end = end_timesteps[i_s - 1]
            t_start = (math.sqrt(1. - rho) * previous_t_end) / (math.sqrt(1. - rho) * previous_t_end + (1. - previous_t_end) * math.sqrt(-rho))
            start_timesteps.append(t_start)
        return start_timesteps

    # Fixed-density allocation

    @staticmethod
    def _normal_cdf(value):
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @staticmethod
    def _logit(value):
        return math.log(value / (1.0 - value))

    def _timestep_cdf(self, t):
        t = min(max(float(t), 0.0), 1.0)
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        z = (self._logit(t) - float(self.logit_mean)) / float(self.logit_std)
        return self._normal_cdf(z)

    def _interval_mass(self, low, high):
        if high <= low:
            return 0.0
        return max(0.0, self._timestep_cdf(high) - self._timestep_cdf(low))

    def _build_fixed_density_bins(self):
        """
        Build stage-allocation probabilities from the global timestep distribution.
        Overlapping intervals share their probability equally between covering stages.

        Returns
            List[dict]: entries with a stage index and normalized probability mass.
                A stage can have multiple entries; all masses sum to 1.
        """
        # Split [0, 1] at each stage's training interval boundaries.
        boundaries = [0.0, 1.0]
        boundaries.extend(float(v) for v in self.start_timesteps)  # Add stage starts
        boundaries.extend(float(v) for v in self.training_end_timesteps)  # Add training lower bounds.
        boundaries = sorted(set(min(max(v, 0.0), 1.0) for v in boundaries))  # Clamp to [0, 1], remove duplicates, and sort.

        bins = []  # One entry for each interval/stage pair.
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            if high <= low:
                continue

            # Stage coverage is constant within each bin; check its midpoint.
            midpoint = 0.5 * (low + high)  # A representative global t in this interval.
            candidates = []  # Stages whose training intervals cover this bin.
            for stage_index, (t_start, training_t_end) in enumerate(zip(self.start_timesteps, self.training_end_timesteps)):
                stage_low = min(t_start, training_t_end)
                stage_high = max(t_start, training_t_end)
                if stage_low <= midpoint <= stage_high:
                    candidates.append(stage_index)

            if not candidates:
                continue

            # Divide this bin's global Logit-Normal probability among covering stages.
            mass = self._interval_mass(low, high)  # P(low <= t <= high), not interval width.
            if mass <= 0.0:
                continue

            split_mass = mass / len(candidates)  # Each covering stage receives an equal share.
            for stage_index in candidates:
                bins.append(
                    {
                        "stage": stage_index,
                        "mass": split_mass,
                    }
                )

        # Normalize over valid training intervals; uncovered bins have no quota.
        total_mass = sum(item["mass"] for item in bins)
        if total_mass <= 0.0:
            raise ValueError("Fixed-density stage allocation produced zero probability mass")

        for item in bins:
            item["mass"] /= total_mass

        return bins

    @staticmethod
    def _apportion_counts(batch_size, masses):
        """
        Convert bin probabilities to integer image quotas.

        Args
            batch_size (int):
                Total number of images to assign.
            masses (List[float]):
                Normalized probability per bin entry.
        Returns
            List[int]: quotas per bin entry, summing to batch_size.
        """
        # Expected image counts may be fractional; first round each down.
        expected = [mass * batch_size for mass in masses]
        counts = [int(math.floor(value)) for value in expected]  # Initial integer quotas per bin entry.
        remaining = batch_size - sum(counts)  # Number of images still unassigned.

        # Sort bin indices by fractional remainder, largest first.
        remainders = sorted(
            range(len(expected)),
            key=lambda i: expected[i] - counts[i],
            reverse=True,
        )

        # Add one image to each of the top remaining entries.
        for i in remainders[:remaining]:
            counts[i] += 1

        return counts

    def _fixed_batch_bin_counts(self, batch_size):
        """
        Get integer image quotas in fixed_density_bins order.

        Args
            batch_size (int):
                Number of images in the batch.
        Returns
            List[int]: one quota per bin entry, summing to batch_size.
        """
        # Keep separate bin entries even when they belong to the same stage.
        masses = [item["mass"] for item in self.fixed_density_bins]  # Preserve bin order.
        return self._apportion_counts(batch_size, masses)

    # Stage-local sampling

    def _sample_stage_local_timesteps(self, count, stage_index, device, dtype):
        """
        Sample noise levels within the selected stage's training interval.

        Args
            count (int):
                Number of images assigned to this bin entry.
            stage_index (int):
                Stage whose interval and Logit-Normal parameters are used.
            device (torch.device):
                Device for sampled tensors.
            dtype (torch.dtype):
                Dtype for returned timesteps.
        Returns
            torch.Tensor: global timesteps [count], between the training lower bound and stage start.
        """
        low = float(self.training_end_timesteps[stage_index])  # Lower global timestep allowed during training.
        high = float(self.start_timesteps[stage_index])  # Stage start in global timestep.

        # Standard normal -> stage-specific normal -> sigmoid values in (0, 1).
        normal = torch.randn(count, device=device, dtype=torch.float64)  # [count]
        # u is relative to the truncated training interval. With a lower bound,
        # the full-stage sigma is min_stage_sigma + (1 - min_stage_sigma) * u.
        u = torch.sigmoid(
            float(self.stage_logit_means[stage_index])
            + float(self.stage_logit_stds[stage_index]) * normal
        )

        # Map the relative position u to the stage's global-timestep training interval.
        timesteps = low + (high - low) * u  # [count]
        return timesteps.clamp(min=low, max=high).to(dtype=dtype)

    def _sample_fixed_batch_timesteps_and_stages(
        self,
        batch_size,
        device,
        dtype,
    ):
        """
        Randomly assign images to fixed quotas, then sample their global timesteps.

        Args
            batch_size (int):
                Number of images in the batch.
            device (torch.device):
                Device for sampled tensors.
            dtype (torch.dtype):
                Dtype for returned timesteps.
        Returns
            tuple: global timesteps [B] and stage indices [B], in original image order.
        """
        counts = self._fixed_batch_bin_counts(batch_size)  # Integer quota per bin entry; sum = B.
        sampled_timesteps = torch.empty(batch_size, device=device, dtype=dtype)  # [B], global t per image.
        chosen_stages = torch.empty(batch_size, device=device, dtype=torch.long)  # [B], stage index per image.

        # Shuffle image indices so stage assignment does not depend on batch order.
        permutation = torch.randperm(batch_size, device=device)

        offset = 0  # Number of indices already taken from permutation.
        for bin_info, count in zip(self.fixed_density_bins, counts):
            if count == 0:
                continue
            indices = permutation[offset:offset + count]  # [count], original image indices for this quota.
            stage_index = int(bin_info["stage"])

            # Bin masses control counts; sample t over the stage's full training interval.
            sampled_timesteps[indices] = self._sample_stage_local_timesteps(
                count,
                stage_index,
                device,
                dtype,
            )
            chosen_stages[indices] = stage_index  # Write results back in original batch order.
            offset += count

        return sampled_timesteps, chosen_stages

    # Hierarchical images and noise

    @torch.no_grad()
    def get_hierarchical_images(self, x):
        """
        Build `[low_res, ..., high_res]` image tensors from a clean image batch.

        Args
            x (torch.Tensor):
                Clean full-resolution images [B, C, H, W].
        Returns
            List[torch.Tensor]: clean images [B, C, H_stage, W_stage], from low to high resolution.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got shape {tuple(x.shape)}")

        # Collect images from high to low resolution, then reverse to stage order.
        image_list = [x]
        height, width = x.shape[-2], x.shape[-1]
        for _ in range(len(self.stages) - 1):
            height //= 2
            width //= 2
            # Halve the image height and width for the next lower-resolution stage.
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
            image_list.append(x)

        image_list = list(reversed(image_list))
        return image_list

    @torch.no_grad()
    def add_continuous_hierarchical_noise(self, images_list):
        """
        Construct noisy inputs and endpoint targets for the assigned images at each stage.

        Args
            images_list (List[torch.Tensor]):
                Clean images [B, C, H_stage, W_stage], ordered from low to high resolution.
                Every stage keeps the same original batch order.
        Returns
            HierarchicalImageNoiseOutput: per-active-stage noisy inputs, sigmas,
                global timesteps, endpoint targets, and original image indices.
        """
        stages = len(self.stages)
        if stages != len(images_list):
            raise ValueError("Number of clean hierarchical images must match len(stages)")

        base_image = images_list[-1]  # [B, C, H, W], the full-resolution clean images.
        device = base_image.device
        dtype = base_image.dtype
        batch_size = base_image.shape[0]

        # Scale full-resolution Gaussian noise, then downsample it for each stage.
        # Multiplying by 2 after downsampling restores the noise variance.
        noise = torch.randn_like(base_image) * self.noise_scale  # [B, C, H, W]
        height, width = noise.shape[-2], noise.shape[-1]
        noise_list = [noise]
        cur_noise = noise
        for _ in range(stages - 1):
            height //= 2
            width //= 2
            cur_noise = F.interpolate(
                cur_noise,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ) * 2.0
            noise_list.append(cur_noise)
        noise_list = list(reversed(noise_list))  # Match images_list's low-to-high resolution order.

        # Sample only where the stage-relative sigma is at least min_stage_sigma.
        sampled_timesteps, chosen_stages = self._sample_fixed_batch_timesteps_and_stages(
            batch_size,
            device,
            dtype,
        )

        stage_to_indices = [
            torch.nonzero(chosen_stages == i_s, as_tuple=False).flatten()
            for i_s in range(stages)
        ]  # stage_to_indices[i] contains the original indices assigned to stage i.

        # One tensor per active stage; stages with no assigned images are omitted.
        noisy_images_list = []
        sigmas_list = []
        targets_list = []
        timesteps_list = []
        sample_indices_list = []

        for i_s, indices in enumerate(stage_to_indices):
            # Small batches can leave some stages empty.
            if indices.numel() == 0:
                continue

            # Gather matching images, noise, and timesteps using original batch indices.
            clean_image = images_list[i_s].index_select(0, indices)  # [B_stage, C, H_stage, W_stage]
            stage_noise = noise_list[i_s].index_select(0, indices)  # Same shape as clean_image.
            timesteps = sampled_timesteps.index_select(0, indices)  # [B_stage], global t.
            t_start = torch.as_tensor(self.start_timesteps[i_s], device=device, dtype=dtype)
            t_end = torch.as_tensor(self.end_timesteps[i_s], device=device, dtype=dtype)

            # Stage-relative sigma is 1 at t_start and 0 at t_end.
            denom = (t_start - t_end).clamp_min(torch.finfo(dtype).eps)
            sigma = ((timesteps - t_end) / denom).clamp(0.0, 1.0)  # [B_stage]

            # Construct the start image at global t_start.
            if i_s == 0:
                x_start = stage_noise  # First stage starts from pure noise.
            else:
                # Use the same images at the previous resolution, upsampled to this stage.
                last_clean_image = images_list[i_s - 1].index_select(0, indices)
                last_clean_image = F.interpolate(
                    last_clean_image,
                    size=(last_clean_image.shape[-2] * 2, last_clean_image.shape[-1] * 2),
                    mode="nearest",
                )
                x_start = t_start * stage_noise + (1.0 - t_start) * last_clean_image

            # Construct the endpoint the model must predict.
            if i_s == stages - 1:
                x_end = clean_image  # Final endpoint is clean, at t=0 in our configuration.
            else:
                x_end = t_end * stage_noise + (1.0 - t_end) * clean_image  # Intermediate endpoint retains noise.

            # Broadcast sigma from [B_stage] to [B_stage, 1, 1, 1].
            while sigma.ndim < x_start.ndim:
                sigma = sigma.unsqueeze(-1)

            # Interpolate from the stage's start image toward its endpoint target.
            noisy_images = sigma * x_start + (1.0 - sigma) * x_end  # [B_stage, C, H_stage, W_stage]

            noisy_images_list.append(noisy_images.to(dtype))
            sigmas_list.append(sigma.to(dtype))  # [B_stage, 1, 1, 1], used by the loss.
            timesteps_list.append(timesteps.to(dtype))  # [B_stage], used to condition the model.
            targets_list.append(x_end.to(dtype))
            sample_indices_list.append(indices)  # [B_stage], used to select matching class labels.

        return HierarchicalImageNoiseOutput(
            noisy_images_list=noisy_images_list,
            sigmas_list=sigmas_list,
            timesteps_list=timesteps_list,
            targets_list=targets_list,
            sample_indices_list=sample_indices_list,
        )
