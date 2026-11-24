"""Masked diffusion velocity objective for the true MANO DiT head."""

import torch


PER_HAND = 109


class ManoDiffusionLoss:
    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.weight = float(cfg.get("weight", 1.0))

    def __call__(self, pred, batch):
        velocity_pred = pred.get("_mano_diffusion_velocity_pred")
        velocity_target = pred.get("_mano_diffusion_velocity_target")
        if velocity_pred is None or velocity_target is None:
            zero = pred["hand"].sum() * 0.0
            return zero, {"metric/mano_diffusion/skipped": zero.detach() + 1.0}

        error = (
            velocity_pred.float().reshape(*velocity_pred.shape[:-1], 2, PER_HAND)
            - velocity_target.float().reshape(*velocity_target.shape[:-1], 2, PER_HAND)
        ).square()
        mask = batch["hand_kept"].to(device=error.device).float()
        valid = batch.get("mano_gt_valid")
        if valid is not None:
            mask = mask * valid.to(device=error.device).float().reshape(-1, 1, 1)
        expanded_mask = mask[..., None].expand_as(error)
        loss = torch.where(
            expanded_mask > 0, error, torch.zeros_like(error)
        ).sum() / expanded_mask.sum().clamp(min=1.0)
        timestep = pred.get("_mano_diffusion_timestep")
        logs = {"loss/mano_diffusion/velocity_mse": loss.detach()}
        if timestep is not None:
            logs["metric/mano_diffusion/timestep_mean"] = timestep.float().mean().detach()
        return self.weight * loss, logs
