import math

def adjust_learning_rate(optimizer, epoch, args):
    """Set learning rate at a fractional epoch (epoch + iteration progress).

    Schedules:
        linear: After warmup, decrease LR along a straight line from args.lr
            to args.min_lr at decay_end_epoch; then hold args.min_lr.
            During warmup, hold args.lr.
        constant: Linearly increase LR from zero to args.lr during warmup,
            then keep args.lr for the rest of training. No decay to min_lr.
        cosine_tail: Use the same warmup as constant, then hold args.lr until
            the final args.lr_decay_epochs before decay_end_epoch. In that
            window, use a half-cosine curve to decrease smoothly to min_lr,
            then hold min_lr. Decay never starts before warmup finishes.

    decay_end_epoch is args.lr_decay_end_epoch when supplied, otherwise
    args.epochs. Parameter groups with lr_scale receive LR * lr_scale.
    """
    # Linear scheduling holds base LR during warmup.
    if args.lr_schedule == "linear":
        decay_end_epoch = args.lr_decay_end_epoch if args.lr_decay_end_epoch is not None else args.epochs
        decay_end_epoch = max(decay_end_epoch, args.warmup_epochs)
        if epoch < args.warmup_epochs:
            lr = args.lr
        elif epoch >= decay_end_epoch:
            lr = args.min_lr
        else:
            progress = (epoch - args.warmup_epochs) / max(decay_end_epoch - args.warmup_epochs, 1e-8)
            lr = args.lr - (args.lr - args.min_lr) * progress

    # Constant and cosine-tail schedules share a linear warmup from zero.
    elif epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        
        # Constant: keep the base LR after warm-up.
        if args.lr_schedule == "constant":
            lr = args.lr

        # Cosine tail: only the final decay window reduces the base LR.
        elif args.lr_schedule == "cosine_tail":
            decay_end_epoch = args.lr_decay_end_epoch if args.lr_decay_end_epoch is not None else args.epochs
            decay_epochs = float(args.lr_decay_epochs)
            if decay_epochs <= 0:
                raise ValueError("--lr_decay_epochs must be positive for cosine_tail")
            # Hold base LR until the final decay window, then approach min_lr.
            decay_start_epoch = max(float(args.warmup_epochs), decay_end_epoch - decay_epochs)
            if epoch < decay_start_epoch:
                lr = args.lr
            elif epoch >= decay_end_epoch:
                lr = args.min_lr
            else:
                progress = (epoch - decay_start_epoch) / max(decay_end_epoch - decay_start_epoch, 1e-8)
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * progress))

        else:
            raise NotImplementedError

    # The composite optimizer exposes both Muon and AdamW groups here.
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
