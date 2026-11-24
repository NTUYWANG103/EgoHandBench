"""ViDiHand2 inference: `load_model(weights)` builds the model, `model(sample)` returns per-frame MANO and 21 joints of both hands."""

import math
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from baselines.vidihand_v2.archs.arch_utils import (assemble_reads, depth_pairs, depth_roots, fuse, motion_cues, ray_joints, read_consistency,
                                                    read_cues, read_moments, read_stats, solve_roots)
from baselines.vidihand_v2.archs.vidihand2_arch import (ROPE_MAX_LEN, AmodalHead, Gate, HeatmapReader, ManoHeat, MotionPrecision, Projector,
                                                        ReadEvidence, ReadPrecision, Tower)
from baselines.vidihand_v2.archs.wan import VAEEncoder
from baselines.vidihand_v2.io import model_size, resize as resize_video, tf32
from baselines.vidihand_v2.mano import MIRROR, ManoModel

MODES, DECODES = ("autoregressive", "independent"), ("clip", "per_window")
HEADS = ("projector", "reader", "mano_heat", "amodal", "evidence", "gate", "tprec", "rprec")
Window = namedtuple("Window", "start real frames latents lead")


def load_model(weights, device="cuda"):
    """ViDiHand2 on a CUDA device, with the weights given by path: `vdm` and `projector` (safetensors files) and `mano` (the folder of
    MANO_RIGHT.pkl)."""
    return ViDiHandV2(yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text()), weights, device)


class ViDiHandV2:
    """Video in, hands out: `model(sample)` with a sample dict (see `__call__`). The video is resized inside (by default
    to a 480-pixel short side, see `__call__`), and the outputs are in the input camera, whose intrinsics may change from frame to frame.
    Returns dict(joint=..., mano=...): `joint` holds joints3d [T, 2, 21, 3] (metres, camera frame, hands [left, right], MANO joint order
    then the five fingertips) and valid [T, 2]; `mano` holds global_orient [T, 2, 1, 3, 3], hand_pose [T, 2, 15, 3, 3] (rotation
    matrices, MANO mean pose included), betas [T, 2, 10], cam_trans [T, 2, 3] and valid; values of hands that are not valid are NaN."""

    def __init__(self, cfg, weights, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("ViDiHand2 runs on CUDA only")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        with torch.cuda.device(self.device):
            self.cfg, self.latents = cfg["inference"], cfg["stream_latents"]
            self.root_cfg = SimpleNamespace(**cfg["root"])
            self.root_cfg.pairs = depth_pairs(ManoModel.parents, self.root_cfg.knuckles)
            self.to_mano = torch.as_tensor(np.argsort(ManoModel.openpose), device=self.device)
            dit, vae, tok = cfg["dit"], cfg["vae"], cfg["tokens"]
            vdm = load_file(str(weights["vdm"]))
            part = lambda state, prefix: {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            with torch.device("meta"):
                self.tower = Tower(**dit, z_dim=vae["z_dim"], hidden=cfg["projector"]["hidden"], n_tokens=sum(tok.values()))
            self.tower.load_state_dict(part(vdm, "tower."), strict=True, assign=True)
            self.tower = self.tower.eval().requires_grad_(False).to(self.device)
            with tf32(True):
                self.tower.prepare(vdm["prompt_embedding"])
            self.vae = VAEEncoder(**vae, first=4 * self.latents - 3, cont=4 * self.latents)
            self.vae.load_state_dict(part(vdm, "vae."), strict=True)
            self.vae = self.vae.eval().requires_grad_(False).to(self.device, torch.bfloat16)
            heads = load_file(str(weights["projector"]))
            pc, hidden, th = cfg["projector"], cfg["projector"]["hidden"], cfg["time_decoder"]["hidden"]
            self.projector = Projector(dit["dim"], **pc, **{f"n_{k}": v for k, v in tok.items()}, time_hidden=th)
            self.reader = HeatmapReader(dit["dim"], hidden, **cfg["reader"], time_hidden=th)
            self.mano_heat = ManoHeat(hidden, pc["head_hidden"], **cfg["mano_heat"])
            self.amodal, self.evidence = AmodalHead(hidden), ReadEvidence(hidden)
            self.gate = Gate(hidden, self.root_cfg.ev_clamp)
            acceleration = (self.root_cfg.lateral_acceleration, self.root_cfg.depth_acceleration / math.sqrt(self.root_cfg.couple))
            self.tprec, self.rprec = MotionPrecision(**cfg["tprec"], acceleration=acceleration), ReadPrecision(**cfg["rprec"])
            for name in HEADS:
                module = getattr(self, name)
                module.load_state_dict(part(heads, f"{name}."), strict=True)
                setattr(self, name, module.eval().to(self.device))
            self.mano = ManoModel(weights["mano"], device=self.device)

    def stream_windows(self, n):
        latents = self.latents
        first, cont = 4 * latents - 3, 4 * latents
        real = min(first, n)
        frames = 1 + 4 * -(-(real - 1) // 4)
        out = [Window(0, real, frames, slice(0, 1 + (frames - 1) // 4), 0)]
        for s in range(first, n, cont):
            real = min(cont, n - s)
            frames = 4 * -(-real // 4)
            l0 = latents + (s - first) // 4
            out.append(Window(s, real, frames, slice(l0 - 1, l0 + frames // 4), -1))
        return out

    def read(self, x, lead, cache, ar):
        (feat, special), free, caps = self.tower(x, self.projector.seeds(x.shape[2]), cache, ar)
        free = free or (feat, special)
        d = 1 if lead < 0 else 0
        split = lambda r: ((r[0][:, :, d:], r[1][d:]), (r[0][:, :, :d], r[1][:d]) if d else None)
        return split((feat, special)), split(free), self.tower.next_cache(caps, x.shape[2]) if ar else None

    def window_features(self, prev, own, nxt):
        (kb, ka), blend = self.cfg["context"], float(self.cfg["junction_blend"])
        mix = lambda a, b: tuple((1.0 - blend) * x + blend * y.to(x.dtype) for x, y in zip(a, b))
        cut = lambda r, a, b: (r[0][:, :, a:b], r[1][a:b])
        swap_last = lambda r, j: (torch.cat([r[0][:, :, :-1], j[0]], 2), torch.cat([r[1][:-1], j[1]], 0))
        parts, mid, lead = [], own[0], 0
        if prev is not None:
            parts.append(swap_last(cut(prev[0], -kb, None), mix(cut(prev[0], -1, None), own[1])))
            lead = 3 - 4 * kb
        if nxt is not None:
            mid = swap_last(mid, mix(cut(own[0], -1, None), nxt[1]))
        parts.append(mid)
        if nxt is not None:
            parts.append(cut(nxt[0], 0, ka))
        return torch.cat([q[0] for q in parts], 2), torch.cat([q[1] for q in parts], 0), lead

    def decode(self, feats, free, rgb, start, camera, canvas):
        feat, special, lead = feats
        n = len(rgb)
        intr = [tuple(map(float, row)) for row in camera(start + lead + 4 * np.arange(feat.shape[2]))]
        p = self.projector(feat, special, n, lead, intr, canvas)
        tok_px, tok_var = read_moments(p["joints2d"], p["joints2d_m2"], canvas)
        fine = self.reader(rgb, feat, p["joint_feat"].reshape(n, 42, -1), (tok_px.reshape(n, -1, 2), tok_var.reshape(n, -1)), lead)
        if free is not None:
            p.update(self.projector.hand_states(free[0], free[1], n, free[2], intr, canvas))
        return dict(p, **fine)

    @torch.no_grad()
    @tf32(True)
    def predict(self, video, K, fps, mode, decode="clip"):
        n, height, width = video.shape[:3]
        if n < 2:
            raise ValueError("ViDiHand2 needs a video of at least 2 frames")
        if height % 32 or width % 32:
            raise ValueError("ViDiHand2 needs an RGB canvas whose sides are multiples of 32")
        if np.allclose(K, K[:1], rtol=0.0, atol=1e-6):
            K = K[:1]
        elif len(K) != n:
            raise ValueError(f"ViDiHand2 needs one camera, or one per frame: got {len(K)} for {n} frames")
        with torch.cuda.device(self.device):
            k = np.asarray(K, np.float32)
            intr = np.stack((k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]), 1)
            camera = lambda ids: intr[np.clip(ids, 0, len(intr) - 1)]
            canvas = (width, height)
            frames = lambda a, m: video[np.minimum(np.arange(a, a + m), n - 1)]
            rgb = lambda a, m: torch.as_tensor(frames(a, m), device=self.device)
            ar = mode == "autoregressive"
            windows = self.stream_windows(n)
            padded = 1 + 4 * -(-(n - 1) // 4)
            with tqdm(total=padded, desc="ViDiHand2: VAE encoding", unit="frame") as bar:
                latent = self.vae(frames(0, padded), bar)
            if decode == "per_window":
                return self.predict_windows(latent, windows, rgb, camera, canvas, fps, ar)
            bar = tqdm(total=len(windows) + 1, desc="ViDiHand2: reconstructing hands", unit="step")
            reads, cache = {}, None

            def read(i):
                nonlocal cache
                if i not in reads:
                    reads[i] = self.read(latent[:, windows[i].latents][None].to(self.device).float(), windows[i].lead, cache, ar)
                    cache = reads[i][2]
                return reads[i]

            records = []
            for i, (s, _, m, _, _) in enumerate(windows):
                for j in range(i, min(i + 2, len(windows))):
                    read(j)
                near = lambda j, b: reads[j][b] if 0 <= j < len(windows) else None
                feats = self.window_features(near(i - 1, 0), reads[i][0], near(i + 1, 0))
                free = self.window_features(near(i - 1, 1), reads[i][1], near(i + 1, 1)) if ar else None
                records.append(self.decode(feats, free, rgb(s, m), s, camera, canvas))
                reads.pop(i - 1, None)
                bar.update()
            reads.clear()
            rec = {key: torch.cat([r[key] for r in records]) for key in records[0]}
            out = self.solve(rec, camera(np.concatenate([np.arange(w.start, w.start + w.frames) for w in windows])), fps, canvas)
            bar.update()
            bar.close()
            return {key: v[:n].float().cpu().numpy() for key, v in out.items()}

    def predict_windows(self, latent, windows, rgb, camera, canvas, fps, ar):
        bar = tqdm(total=len(windows), desc="ViDiHand2: reconstructing hands", unit="window")
        outs, cache = [], None
        for s, real, m, sl, lead in windows:
            x = latent[:, sl][None].to(self.device).float()
            (feat, special), _, caps = self.tower(x, self.projector.seeds(x.shape[2]), cache, ar, fork=False)
            cache = self.tower.next_cache(caps, x.shape[2]) if ar else None
            rec = self.decode((feat, special, lead), None, rgb(s, m), s, camera, canvas)
            out = self.solve(rec, camera(np.arange(s, s + m)), fps, canvas)
            outs.append({key: v[:real] for key, v in out.items()})
            bar.update()
        bar.close()
        return {key: torch.cat([o[key] for o in outs]).float().cpu().numpy() for key in outs[0]}

    def rel_joints(self, betas, R, L, right):
        rotations = torch.cat((R.reshape(-1, 1, 3, 3), L.reshape(-1, 15, 3, 3)), 1).float()
        with tf32(False):
            out = self.mano.layer(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:], betas=betas)
        joints = torch.cat((out.joints[:, :16], out.vertices[:, list(self.mano.tip_vertices)]), 1)
        if not right:
            joints = joints * torch.as_tensor(MIRROR, device=joints.device)
        joints = joints.float()
        return joints - joints[:, :1]

    def solve(self, rec, intr, fps, canvas):
        c, inv, t = self.root_cfg, self.to_mano, len(rec["exists_3d"])
        intr4 = torch.as_tensor(intr, device=self.device)
        R, L = rec["global_orient"].float(), rec["hand_pose"].float()
        betas = rec["frame_betas"].median(0).values[None].expand(t, 2, 10).float().mean(0)
        R, L, betas = self.mano_heat(rec["hand_feat"], rec["shape_tokens"], rec["exists_3d"], R, L, betas)
        rel = torch.stack([self.rel_joints(betas[s][None].expand(t, -1), R[:, s], L[:, s], s == 1) for s in (0, 1)], 1)

        scale = torch.tensor([float(canvas[0]), float(canvas[1])], device=self.device)
        mu = rec["joints2d"].float()
        am_px = (self.amodal(rec["joint_feat"], mu) * scale)[:, :, inv]
        _, tvar = read_moments(mu[:, :, inv], rec["joints2d_m2"][:, :, inv], canvas)
        fine = (rec["read_px"].reshape(t, 2, 21, 2), rec["var_px"].reshape(t, 2, 21), rec["p_null"].reshape(t, 2, 21), rec["win_mass"].reshape(t, 2, 21))
        e = torch.sigmoid(self.evidence(rec["joint_feat"], read_stats(*fine, canvas, c.ev_clamp)).float())[:, :, inv]
        read, var, alpha = assemble_reads(am_px, canvas, [x[:, :, inv] for x in fine], tvar, c)
        w = rec["exists_3d"].float()[..., None].expand(t, 2, 21).contiguous()
        omega, ray, s3 = read_consistency(rel, read, var, w, intr4, c)
        root = depth_roots(rel, read, var, w, intr4, fps, omega, ray, c)
        claim = rec["exists_2d"].float() > self.cfg["claim"]
        p_null = fine[2][:, :, inv]
        log_sig = self.tprec(motion_cues(read, e, claim, alpha, p_null, fps, c.pairs))
        h = self.rprec(read_cues(read, var, e, alpha, p_null, s3, canvas))
        root, unplaced = solve_roots(rel, read, var, alpha, e, claim, intr4, fps, root, log_sig.double().exp(), h.double(),
                                     self.rprec.sigma(), c)
        root = torch.where(unplaced[..., None], torch.full_like(root, float("nan")), root)
        pooled = rec["pooled_px"].reshape(t, 2, 21, 2)[:, :, inv]
        on_rays = ray_joints(alpha[..., None] * pooled + (1 - alpha[..., None]) * am_px, root[..., 2], rel, intr4)
        e2 = rec["exists_2d"].float().clamp(1e-05, 1 - 1e-05)
        g, part_in = self.gate(rec["joint_feat"][:, :, inv], e, e2)
        joints = fuse(on_rays, rel + root[:, :, None], g[..., None], self.gate.parts, part_in)
        joints = torch.where((joints[..., 2] > 0).all(-1)[..., None, None], joints, torch.full_like(joints, float("nan")))
        return dict(joints3d=joints, presence=torch.sigmoid(torch.log(e2 / (1 - e2))))

    def __call__(self, sample, mode="autoregressive", decode="clip", fit_mano=True, resize=480):
        """Per-frame joints and their MANO fit (skipped with `fit_mano=False`) in the input camera; the video runs at a short side of
        `resize` pixels (None: its own size), sides rounded to multiples of 32."""
        if mode not in MODES or decode not in DECODES:
            raise ValueError(f"mode is one of {MODES} and decode one of {DECODES}, got {mode!r} / {decode!r}")
        canvas = model_size(sample["video"].shape[2], sample["video"].shape[1], resize)
        if max(canvas) > 32 * ROPE_MAX_LEN:
            raise ValueError(f"ViDiHand2 runs on canvases of up to {32 * ROPE_MAX_LEN} pixels a side, got {canvas[0]} x {canvas[1]}: lower resize")
        video, K = resize_video(sample["video"], sample["K"], canvas)
        raw = self.predict(video, K, sample["fps"], mode, decode)
        valid = (raw["presence"] > self.cfg["claim"]) & np.isfinite(raw["joints3d"]).all((-1, -2))
        joints = np.where(valid[..., None, None], raw["joints3d"], np.nan).astype(np.float32)
        ids = np.asarray(sample["source_frame_ids"])
        joint = dict(joints3d=joints, valid=valid, source_frame_ids=ids)
        if not fit_mano:
            return dict(mano=None, joint=joint)
        with torch.cuda.device(self.device):
            mano = dict(self.mano.joint2mano(joints, valid), source_frame_ids=ids)
        return dict(mano=mano, joint=joint)
