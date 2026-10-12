import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

from util.crop import build_train_transform
import util.misc as misc
from util.muon import GMuonWithAdamW

import copy
from engine_pixelmoe import train_one_epoch, evaluate

from denoiser import Denoiser
from model_pixelmoe import PixelMoE_models


def get_args_parser():
    parser = argparse.ArgumentParser('PixelMoE', add_help=False)

    # architecture
    parser.add_argument('--model', default='PixelMoE-B-E32A4/16', type=str, metavar='MODEL',
                        choices=tuple(PixelMoE_models),
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument(
        '--flux_attn_proj_dropout',
        action='store_true',
        help='Apply --proj_dropout to FLUX attention output projections',
    )
    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR at the end of decay')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine_tail'],
                        help='Learning rate schedule')
    parser.add_argument('--lr_decay_end_epoch', type=float, default=None,
                        help='Epoch where linear or cosine_tail decay reaches min_lr. Defaults to --epochs.')
    parser.add_argument('--lr_decay_epochs', type=float, default=40.0,
                        help='For cosine_tail, number of epochs spent decaying to min_lr (default: 40).')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Clip the global gradient norm before optimizer.step; <= 0 disables clipping')
    parser.add_argument('--warmup_max_grad_norm', type=float, default=None,
                        help='Gradient clipping limit during LR warmup. Defaults to --max_grad_norm; <= 0 disables clipping during warmup.')
    parser.add_argument('--grad_clip_start_epoch', type=float, default=0.0,
                        help='Epoch progress at which gradient clipping starts (default: 0).')
    parser.add_argument('--skip_grad_norm_threshold', type=float, default=0.0,
                        help='Skip the entire optimizer/router/EMA update when the pre-clipping global gradient norm exceeds this value or is non-finite; <= 0 disables skipping')
    parser.add_argument('--skip_grad_norm_start_epoch', type=float, default=None,
                        help='Epoch progress at which large-gradient skipping starts. Defaults to --warmup_epochs.')
    parser.add_argument('--max_skipped_grad_steps_per_epoch', type=int, default=0,
                        help='Abort after more than this many large-gradient optimizer updates are skipped in one epoch; <= 0 disables the limit')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--use_second_ema', action='store_true',
                        help='Also track and save the second EMA. Disabled by default to reduce per-step overhead.')
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--sigma_min', default=2e-2, type=float,
                        help='Lower bound on stage-relative sigma in the endpoint-loss and velocity denominators')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--use_flash_attn', action='store_true',
                        help='Use flash-attn for FLUX-style variable-length attention')
    parser.add_argument('--hierarchical_stages', nargs='+', default=[0, 1], type=int)
    parser.add_argument('--hierarchical_end_timesteps', nargs='+', default=[0.75, 0.0], type=float,
                        help='Per-stage endpoint global timesteps t_end')
    parser.add_argument('--hierarchical_logit_mean', default=0.8, type=float,
                        help='Global Logit-Normal mean for training stage allocation and sampling timestep spacing. Generation scripts use 0.')
    parser.add_argument('--hierarchical_logit_std', default=0.8, type=float,
                        help='Global Logit-Normal std for training stage allocation and sampling timestep spacing. Generation scripts use 1.')
    parser.add_argument('--hierarchical_stage_logit_means', nargs='+', default=None, type=float,
                        help='Per-stage logit-normal means; defaults to 0.6 for each stage.')
    parser.add_argument('--hierarchical_stage_logit_stds', nargs='+', default=None, type=float,
                        help='Per-stage logit-normal stds; defaults to 0.8 for each stage.')
    parser.add_argument('--hierarchical_noise_rho', default=-1.0 / 3.0, type=float,
                        help='Correlation between pixels in each 2x2 block of transition noise')
    parser.add_argument('--hierarchical_min_stage_sigma', default=0.0, type=float,
                        help='Lower bound on sampled stage-relative sigma.')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE sampling method')
    parser.add_argument('--num_sampling_steps', nargs='+', default=[50], type=int,
                        help='For one value per stage, the values are added to set the global timestep grid density. A single value is multiplied by the number of stages. Each stage uses grid points within its interval.')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--eval_use_ema', action='store_true', default=True,
                        help='Use EMA weights during generation/evaluation')
    parser.add_argument('--no_eval_ema', action='store_false', dest='eval_use_ema',
                        help='Use raw model weights during generation/evaluation')
    parser.add_argument('--eval_use_second_ema', action='store_true',
                        help='Use the second EMA during generation/evaluation. Implies --use_second_ema.')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument(
        '--random_crop_scales',
        type=float,
        nargs='+',
        default=None,
        metavar='SCALE',
        help='Choose an expansion scale, then randomly crop to --img_size; omit for center crop',
    )
    parser.add_argument(
        '--data_noise_std',
        default=0.0,
        type=float,
        help='Standard deviation of Gaussian noise added to training images after normalization to [-1, 1].',
    )
    parser.add_argument('--regularization_start_epoch', default=None, type=int,
                        help='Enable random crop, input noise, and projection dropout from this epoch; omit to apply them throughout training')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs')
    parser.add_argument('--resume', default='',
                        help='Auto-resume folder; start from scratch if no checkpoint-last exists')
    parser.add_argument('--resume_ckpt', default='',
                        help='Checkpoint .pth path to resume from. Overrides --resume')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to update checkpoint-last')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    if args.eval_use_second_ema:
        if not args.eval_use_ema:
            raise ValueError("--eval_use_second_ema cannot be combined with --no_eval_ema")
        # Enable EMA2 buffers before constructing the model.
        args.use_second_ema = True

    misc.init_distributed_mode(args)

    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = build_train_transform(
        img_size=args.img_size,
        random_crop_scales=args.random_crop_scales if args.regularization_start_epoch is None else None,
    )

    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    if global_rank == 0:
        print(f"Training dataset: len={len(dataset_train)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    # GMuon for matrix/expert weights, AdamW for the remaining parameter groups.
    optimizer = GMuonWithAdamW(
        model.named_parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        muon_momentum=0.95,
        muon_nesterov=True,
        ns_coefficients_preset="POLAR_EXPRESS_COEFFICIENTS",
        adjust_lr="rms_norm",
        skip_keywords=("adaLN_modulation", "final_layer", "ddt_final_layer", "t_embedder", "y_embedder"),
    )
    optimizer._muon.newton_schulz = torch.compile(
        optimizer._muon.newton_schulz,
        dynamic=False,
        fullgraph=False,
        mode="default",
    )
    if misc.is_main_process():
        print("Using GMuonWithAdamW (adjust_lr=rms_norm)")
        print(
            "Compiled GMuon Newton-Schulz core with torch.compile "
            "(mode=default, dynamic=False, no CUDA graphs)"
        )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=True,
    )
    model_without_ddp = model.module

    # Resume from checkpoint if provided.
    checkpoint_path = args.resume_ckpt or (
        os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    )
    checkpoint_exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
    if checkpoint_exists:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        if args.eval_use_second_ema and 'model_ema2' not in checkpoint:
            raise KeyError(
                "--eval_use_second_ema was requested, but the checkpoint does not contain model_ema2"
            )
        ema_state_dict2 = checkpoint['model_ema2'] if args.use_second_ema else None
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        if args.use_second_ema:
            model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        # Restore EMA router_bias buffers separately from trainable parameters.
        model_without_ddp.init_ema_buffers(ema_state_dict1, ema_state_dict2)
        print("Resumed checkpoint from", checkpoint_path)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        if args.use_second_ema:
            model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.init_ema_buffers()
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        if not checkpoint_exists:
            raise ValueError("Generation evaluation requires an existing checkpoint")
        print("Evaluating checkpoint from", checkpoint_path)
        evaluation_rng_devices = (
            [torch.cuda.current_device()] if device.type == "cuda" else []
        )
        print(
            f"Evaluating cfg={args.cfg}, interval_max={args.interval_max}, interval_min={args.interval_min}: {args.output_dir}",
            flush=True,
        )
        with torch.random.fork_rng(devices=evaluation_rng_devices):
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, batch_size=args.gen_bsz,
                        log_writer=log_writer, epoch=args.start_epoch)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    dropout_settings = [
        (module, module.p) for module in model_without_ddp.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    for epoch in range(args.start_epoch, args.epochs):
        if args.regularization_start_epoch is not None:
            # Set the initial/resumed state, then switch at the configured epoch.
            if epoch == args.start_epoch or epoch == args.regularization_start_epoch:
                active = epoch >= args.regularization_start_epoch
                # Workers restart each epoch and pick up the new transform.
                dataset_train.transform = build_train_transform(
                    img_size=args.img_size,
                    random_crop_scales=args.random_crop_scales if active else None,
                )
                for dropout, probability in dropout_settings:
                    dropout.p = probability if active else 0.0
                print(f"Training regularization {'enabled' if active else 'disabled'} at epoch {epoch}")

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically.
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )
        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
