"""MANO utility functions: rotation conversions, forward passes, initialization.

Single canonical home for all MANO helpers + MANO conventions.

Conventions (single source of truth — used by the projector, models/ and viz):
  MANO = the EgoHandBench right-mirror convention: MANO_RIGHT for
                    both slots, full local rotmats, left output x-mirrored.
  N_JOINTS = 16  — MANO native (matches smplx output.joints):
                    0=wrist, 1-3=index, 4-6=middle, 7-9=pinky,
                    10-12=ring, 13-15=thumb. NO fingertip vertices.
  N_SLOTS  = 2   — left + right hand. Slot 0 = left, slot 1 = right.
"""
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F


# MANO conventions (constants used across the codebase).
N_JOINTS: int = 16
N_SLOTS: int = 2

# MANO model files are external licensed DATA (not redistributed here). The
# adapter sets ACE_EGO_HAND_MANO_DIR to the directory holding MANO_RIGHT.pkl
# before importing this module; there is no in-repo default.
MANO_CHECKPOINT_DIR = os.environ.get("ACE_EGO_HAND_MANO_DIR")


# ── Rotation conversions ──────────────────────────────────────────────

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args: x (B, 6) or any shape ending in 6
    Returns: (B, 3, 3) rotation matrices
    """
    orig_dtype = x.dtype
    x = x.float()
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).to(orig_dtype)




# ── MANO forward ──────────────────────────────────────────────────────

# 5 finger-tip vertex indices in the 778-vertex MANO mesh (OpenPose order).
_MANO_TIP_IDS = [744, 320, 443, 554, 671]
# Remap smplx's 16-joint output + 5 tips → 21 OpenPose joints.
_MANO_TO_OP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]




class RightMirrorFullRotationMano(torch.nn.Module):
    """Frozen RIGHT MANOLayer for both slots; the LEFT slot reflects its output.

    The EgoHandBench right-mirror convention: local
    rotation matrices already include the pose mean and go straight to LBS,
    left = diag(-1, 1, 1) * MANO_RIGHT(...), camera = that + cam_trans.
    ``side`` is 0 (left) or 1 (right).
    """

    def __init__(self, mano_dir, side: int):
        super().__init__()
        import inspect

        import numpy as np
        import smplx

        # The licensed pickle's legacy chumpy objects require these aliases.
        for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                                object=object, unicode=str, str=str).items():
            if name not in np.__dict__:
                setattr(np, name, value)
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        if not os.path.isfile(os.path.join(str(mano_dir), "MANO_RIGHT.pkl")):
            raise FileNotFoundError(f"MANO_RIGHT.pkl not found in {mano_dir}")
        self.side = int(side)
        self.layer = smplx.MANOLayer(str(mano_dir), is_rhand=True, num_pca_comps=45).eval()
        for param in self.layer.parameters():
            param.requires_grad_(False)
        self.register_buffer("output_sign", torch.tensor([-1.0 if side == 0 else 1.0, 1.0, 1.0]))

    def forward_rotmats(self, go, hp, betas):
        n = int(betas.shape[0])
        with torch.autocast(device_type=self.output_sign.device.type, enabled=False):
            out = self.layer(global_orient=go.float().reshape(n, 1, 3, 3),
                             hand_pose=hp.float().reshape(n, 15, 3, 3), betas=betas.float())
            return SimpleNamespace(vertices=out.vertices * self.output_sign,
                                   joints=out.joints[:, :16] * self.output_sign)


def mano_forward_batch_full(go_rotmat, hp_rotmat, betas, mano_model):
    """Batched differentiable MANO forward, returns ``(joints_21, vertices_778)``.

    Args:
        go_rotmat: (B, 3, 3) or (B, 1, 3, 3) global orient rotmats
        hp_rotmat: (B, 15, 3, 3) hand-pose rotmats
        betas:     (B, 10)
        mano_model: a RightMirrorFullRotationMano on the same device as the inputs.

    Returns:
        ``(joints_21, vertices_778)`` torch tensors on the model device.
    """
    out = mano_model.forward_rotmats(go_rotmat, hp_rotmat, betas)
    tips = out.vertices[:, _MANO_TIP_IDS]                                    # (B, 5, 3)
    joints_21 = torch.cat([out.joints, tips], dim=1)[:, _MANO_TO_OP]         # (B, 21, 3)
    return joints_21, out.vertices


def setup_mano_models(opt: dict, device: torch.device):
    if not MANO_CHECKPOINT_DIR:
        raise RuntimeError("Set ACE_EGO_HAND_MANO_DIR to the directory holding MANO_RIGHT.pkl")
    models = {bool(side): RightMirrorFullRotationMano(MANO_CHECKPOINT_DIR, side).to(device).eval()
              for side in (0, 1)}
    print(f"Loaded full-rotation RIGHT/mirror MANO on {device} for the projector's translation decode")
    return models
