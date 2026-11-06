"""Native PAD model construction and unchanged temporal refinement."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import numpy as np
from tqdm import tqdm
from baselines.pad_hand.models.pad_hand import PAD_Hand
from baselines.pad_hand.models.MANO import MANO
from baselines.pad_hand.models.difussionsampler import DiffusionSampler
from baselines.pad_hand.models.utils import rotmat2eulerzxy, batch_euler2matzxy

def load_pad_hand_model(checkpoint_path, device):
    model = PAD_Hand(
        device=device, seqlen=16, decoder=True,
        d_model=512, nhead=8, d_hid=512, nlayers=4, dropout=0.1,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.mode = 'test'
    return model

def refine_with_pad_hand(frame_preds, pad_hand_model, mano, device, diff_steps=4, kappa=15/180*np.pi):
    """Run reverse diffusion on sliding windows of length seqlen."""
    seqlen  = 16
    sampler = DiffusionSampler(diff_steps, kappa, device)
    refined = [None] * len(frame_preds)

    for start in tqdm(range(0, len(frame_preds) - seqlen + 1, seqlen), desc='PAD_Hand refinement'):
        seq = frame_preds[start:start + seqlen]
        if any(s is None for s in seq):
            continue
        rotmats   = [torch.cat([fd['global_orient'], fd['hand_pose']], dim=0) for fd in seq]
        rotmats_t = torch.stack(rotmats).to(device)                           # (T, 16, 3, 3)
        betas_t   = torch.stack([fd['betas'] for fd in seq]).to(device)       # (T, 10)

        y0_pose  = rotmat2eulerzxy(rotmats_t.reshape(-1, 3, 3)).reshape(1, seqlen, 48).to(device)
        y0_shape = betas_t.unsqueeze(0).to(device)                            # (1, T, 10)
        y0_shape_t = y0_shape.reshape(-1, 10)
        y0_rotmat  = batch_euler2matzxy(y0_pose.reshape(-1, 3))[0].reshape(-1, 16, 3, 3)
        y0_mano_t  = mano.forward(betas=y0_shape_t, thetas_rotmat=y0_rotmat)
        y0_verts   = y0_mano_t.vertices                                        # (T, 778, 3)

        src_mask     = torch.ones((1, seqlen), dtype=torch.bool, device=device)
        x0_pose_pred = y0_pose.clone()
        with torch.no_grad():
            for n in range(diff_steps, 0, -1):
                if n == diff_steps:
                    n_batch = torch.zeros(1, dtype=torch.float32, device=device) + diff_steps
                    xn_pose = y0_pose + kappa * torch.randn_like(y0_pose).to(device)
                else:
                    n_batch = (torch.zeros(1, device=device) + n).long()
                    xn_pose = sampler.q_sample(x0_pose_pred, y0_pose - x0_pose_pred, n_batch)
                xn_rotmat = batch_euler2matzxy(xn_pose.reshape(-1, 3))[0].reshape(-1, 16, 3, 3)
                xn_mano_t = mano.forward(betas=y0_shape_t, thetas_rotmat=xn_rotmat)
                _, decoder_pred = pad_hand_model.forward(
                    y0_verts.reshape(1, seqlen, 778, 3),
                    xn_mano_t.vertices.reshape(1, seqlen, 778, 3),
                    n_batch.float().to(device),
                    src_mask,
                    mano,
                    y0_shape_t,
                )
                _, _, _, _, x0_rotmat_decoder, x0_mano_decoder = decoder_pred
                x0_pose_pred = rotmat2eulerzxy(x0_rotmat_decoder.reshape(-1, 3, 3)).reshape(1, seqlen, 48)
        refined_verts = x0_mano_decoder.vertices  # (T, 778, 3)
        is_r = seq[0]['is_right']
        for i, gi in enumerate(range(start, start + seqlen)):
            rv = refined_verts[i].detach().cpu().clone()
            rv[:, 0] = (2 * is_r - 1) * rv[:, 0]
            refined[gi] = {**frame_preds[gi], 'refined_vertices': rv}

    for i, fd in enumerate(frame_preds):
        if refined[i] is None and fd is not None:
            refined[i] = {**fd, 'refined_vertices': fd['vertices']}
    return refined
