import math

import torch


def _param_mse(params, global_sd, device):
    mse = 0.0
    for name_p, p, name_gp, gp in zip(
        params.keys(),
        params.values(),
        global_sd.keys(),
        global_sd.values(),
        strict=False,
    ):
        if name_p != name_gp:
            raise ValueError("Parameter names do not match")
        mse += torch.norm(p.to(device) - gp.to(device), p=2).item() ** 2
    return mse


def calculate_next_lr(
    device_id,
    last_loss,
    model_state_dict,
    avg_model_state_dict,
    optimizer,
    current_lr,
    beta,
    device,
    target_ratio,
    wandb_run=None,
):
    """
    Calculate and update the learning rate for a device/client.

    Args:
        device_id: Device/node/client ID
        last_loss: Last training loss
        model_state_dict: Current model state dict for this device
        avg_model_state_dict: Average/global model state dict
        optimizer: The optimizer for this device
        current_lr: Current learning rate for this device
        beta: Weight for parameter MSE
        device: Device (cpu/cuda)
        target_ratio: Target ratio for learning rate adjustment
        wandb_run: Optional wandb run object for logging

    Returns:
        Updated learning rate
    """

    # Update optimizer learning rate for all param groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    mse = beta * _param_mse(model_state_dict, avg_model_state_dict, device=device)
    ratio = mse / (last_loss + 1e-12)
    err = math.log(ratio / target_ratio)
    s = math.log(2)
    multiplier = math.exp(-s * math.tanh(0.1 * err**3))
    next_lr = current_lr * multiplier

    print(
        f"Device {device_id}: last_loss={last_loss:.4f}, mse={mse:.4f}, "
        f"ratio={ratio:.4f}, lr: {current_lr:.7f} -> {next_lr:.7f}"
    )
    print(f"multiplier: {multiplier:.4f}, err: {err:.4f}")

    next_lr = min(next_lr, 2e-2)

    if wandb_run is not None:
        wandb_run.log({"learning_rate": next_lr})

    return next_lr


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
