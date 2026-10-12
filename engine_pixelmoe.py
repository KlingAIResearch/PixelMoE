import math
import sys
import os

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched


def _evaluation_class_labels(num_images, class_num, padding=0):
    """Build the global label schedule for sampling."""
    num_images = int(num_images)
    class_num = int(class_num)
    padding = int(padding)
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if class_num <= 0:
        raise ValueError("class_num must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    if num_images % class_num != 0:
        raise ValueError(
            "num_images must be divisible by class_num for balanced all-class sampling"
        )
    labels = np.arange(class_num, dtype=np.int64).repeat(num_images // class_num)
    if padding:
        labels = np.pad(labels, (0, padding), constant_values=0)
    return labels


def _save_generated_image_batch(
    sampled_images,
    *,
    start_idx,
    num_images,
    save_folder,
):
    for batch_index in range(sampled_images.size(0)):
        image_id = start_idx + batch_index
        if image_id >= num_images:
            break
        image_path = os.path.join(save_folder, f"{image_id:05d}.png")
        image_tensor = sampled_images[batch_index]
        generated_image = image_tensor.numpy().transpose([1, 2, 0])[:, :, ::-1]
        cv2.imwrite(image_path, generated_image)


def _max_grad_norm_for_progress(args, epoch_progress):
    # No clipping before the start epoch; use the optional warmup limit during warmup, otherwise max_grad_norm.
    grad_clip_start_epoch = float(args.grad_clip_start_epoch)
    if epoch_progress < grad_clip_start_epoch:
        return 0.0
    max_grad_norm = float(args.max_grad_norm)
    warmup_max_grad_norm = args.warmup_max_grad_norm
    if warmup_max_grad_norm is not None and epoch_progress < float(args.warmup_epochs):
        return float(warmup_max_grad_norm)
    return max_grad_norm


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # If the norm exceeds this positive threshold, skip the optimizer, router
    # bias, and EMA updates together. A non-positive value disables this threshold.
    skip_grad_norm_threshold = float(args.skip_grad_norm_threshold)
    # Delay large-gradient skipping during early training; use the end of warmup
    # when no start epoch is specified.
    skip_grad_norm_start_epoch = args.skip_grad_norm_start_epoch
    if skip_grad_norm_start_epoch is None:
        skip_grad_norm_start_epoch = float(args.warmup_epochs)
    else:
        skip_grad_norm_start_epoch = float(skip_grad_norm_start_epoch)
    # Stop training if too many updates are skipped in one epoch. A non-positive
    # limit disables this stop condition.
    max_skipped_grad_steps = int(args.max_skipped_grad_steps_per_epoch)
    skipped_grad_steps = 0

    data_noise_std = float(args.data_noise_std)
    if data_noise_std < 0.0:
        raise ValueError("--data_noise_std must be non-negative")
    if args.regularization_start_epoch is not None and epoch < args.regularization_start_epoch:
        data_noise_std = 0.0
    data_noise_generator = None
    if data_noise_std > 0.0:
        # Keep augmentation noise independent of the global RNG used by model
        # timestep sampling, classifier-free dropout, and other stochastic ops.
        data_noise_generator = torch.Generator(device=device)
        data_noise_generator.manual_seed(
            int(args.seed)
            + 1_000_003 * misc.get_rank()
            + 10_000_019 * int(epoch)
        )

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        epoch_progress = data_iter_step / len(data_loader) + epoch
        
        # Adjust LR for each optimizer update.
        lr_sched.adjust_learning_rate(optimizer, epoch_progress, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        if data_noise_generator is not None:
            data_noise = torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=data_noise_generator,
            )
            x.add_(data_noise, alpha=data_noise_std)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()

        # Choose the clipping limit for this step.
        active_max_grad_norm = _max_grad_norm_for_progress(args, epoch_progress)

        # Enable large-gradient skipping after skip_grad_norm_start_epoch.
        skip_large_grad_active = (
            skip_grad_norm_threshold > 0.0
            and epoch_progress >= skip_grad_norm_start_epoch
        )

        # Clip all parameter gradients using their combined L2 norm.
        # With an infinite limit, finite gradients are measured without clipping.
        # The return value is the norm BEFORE clipping.
        clip_limit = active_max_grad_norm if active_max_grad_norm > 0.0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_limit,
            error_if_nonfinite=False,
        )
        grad_norm_value = float(grad_norm.item())

        # Decide whether to discard the step using the original, pre-clip norm.
        # NaN/Inf always skips the update, regardless of the threshold or epoch.
        grad_nonfinite = not math.isfinite(grad_norm_value)
        grad_step_skipped = grad_nonfinite or (
            skip_large_grad_active
            and grad_norm_value > skip_grad_norm_threshold
        )

        # Skip on all ranks if any rank requests it.
        if skip_large_grad_active and misc.is_dist_avail_and_initialized():
            status_flags = torch.tensor(
                [int(grad_nonfinite), int(grad_step_skipped)],
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(status_flags, op=torch.distributed.ReduceOp.MAX)
            grad_nonfinite = bool(status_flags[0].item())
            grad_step_skipped = bool(status_flags[1].item())

        if grad_step_skipped:
            # Discard gradients without updating the optimizer, router bias, or EMA.
            skipped_grad_steps += 1
            optimizer.zero_grad()

            # This batch's routing counts must not affect the next accepted update.
            clear_router_counts = getattr(
                model_without_ddp,
                "clear_moe_router_load_counts",
                None,
            )
            if clear_router_counts is not None:
                clear_router_counts()

            # Report the discarded step once, from the main process.
            if misc.is_main_process():
                skip_reason = (
                    "non-finite gradient"
                    if grad_nonfinite
                    else f"grad norm above {skip_grad_norm_threshold}"
                )
                print(
                    f"Skipped optimizer update at epoch {epoch}, iter {data_iter_step}: "
                    f"grad_norm={grad_norm_value}, reason={skip_reason} "
                    f"({skipped_grad_steps} skipped this epoch)"
                )

            # Stop if the number of discarded steps exceeds the per-epoch limit.
            if max_skipped_grad_steps > 0 and skipped_grad_steps > max_skipped_grad_steps:
                raise RuntimeError(
                    f"Skipped {skipped_grad_steps} large-gradient updates in epoch {epoch}; "
                    f"limit is {max_skipped_grad_steps}"
                )
        else:
            # Update the optimizer, router bias, and EMA.
            optimizer.step()
            optimizer.zero_grad()

            update_moe_router_bias = getattr(
                model_without_ddp,
                "update_moe_router_bias",
                None,
            )
            if update_moe_router_bias is not None:
                update_moe_router_bias()

            torch.cuda.synchronize()

            model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        # Exclude NaN/Inf from the running average and TensorBoard curve.
        if not grad_nonfinite:
            metric_logger.update(grad_norm=grad_norm_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        tensorboard_log_due = data_iter_step % args.log_freq == 0
        if tensorboard_log_due:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                if not grad_nonfinite:
                    log_writer.add_scalar('train/grad_norm', grad_norm_value, epoch_1000x)


def evaluate(model_without_ddp, args, batch_size=64, log_writer=None, epoch=0):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    node_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_node_main_process = node_local_rank == 0
    global_batch_size = batch_size * world_size
    num_steps = math.ceil(args.num_images / global_batch_size)
    sampling_steps_name = "-".join(str(step) for step in model_without_ddp.sampling_steps)
    ema_name = ("-ema2" if args.eval_use_second_ema else "-ema1") if args.eval_use_ema else "-raw"

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}{}".format(
            model_without_ddp.method, sampling_steps_name, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1],
            args.num_images, args.img_size,
            ema_name,
        )
    )
    print("Save to:", save_folder)
    # Create the output directory once per node.
    if is_node_main_process:
        os.makedirs(save_folder, exist_ok=True)
    torch.distributed.barrier()

    # EMA copies live on the model device; select them once before generation.
    if args.eval_use_ema:
        ema_params = model_without_ddp.ema_params2 if args.eval_use_second_ema else model_without_ddp.ema_params1
        ema_buffers = model_without_ddp.ema_buffers2 if args.eval_use_second_ema else model_without_ddp.ema_buffers1
        with torch.no_grad():
            for parameter, ema_parameter in zip(model_without_ddp.parameters(), ema_params):
                parameter.copy_(ema_parameter)
            buffers = dict(model_without_ddp.named_buffers())
            for name, ema_buffer in zip(model_without_ddp.ema_buffer_names, ema_buffers):
                buffers[name].copy_(ema_buffer)
        print("Generate with EMA2" if args.eval_use_second_ema else "Generate with EMA1")
    else:
        print("Generate with raw model weights")

    # Give every class the same number of labels across all ranks. Add class-0
    # padding for the final full batch; only the requested images are saved.
    class_label_gen_world = _evaluation_class_labels(
        args.num_images,
        args.class_num,
        padding=global_batch_size,
    )
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        # Seed each generated batch using its first image's global index.
        torch.manual_seed(int(args.seed) + start_idx)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        sampled_images = (
            ((sampled_images + 1.) / 2.)
            .detach()
            .float()
            .mul_(255)
            .round_()
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .cpu()
        )
        _save_generated_image_batch(
            sampled_images,
            start_idx=start_idx,
            num_images=args.num_images,
            save_folder=save_folder,
        )

    torch.distributed.barrier()

    if misc.is_main_process():
        import torch_fidelity

        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/pixelmoe_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/pixelmoe_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    torch.distributed.barrier()
