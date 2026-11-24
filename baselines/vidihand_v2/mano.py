"""The MANO hand model: joints, meshes and the batched joint-to-MANO fit (both hands use the right model, the left mirrored in x)."""

import contextlib
import io
import math

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from baselines.vidihand_v2.io import tf32

# Joint groups in the MANO order (wrist, index, middle, pinky, ring, thumb, then the thumb, index, middle, ring and pinky tips):
# each finger as (base, middle, distal joint, tip) -- MCP, PIP, DIP for index to ring, CMC, MCP, IP for the thumb -- in the order index,
# middle, pinky, ring, thumb; the palm = the wrist and the five finger-base joints.
FINGERS = ((1, 2, 3, 17), (4, 5, 6, 18), (7, 8, 9, 20), (10, 11, 12, 19), (13, 14, 15, 16))
PALM = (0, 1, 4, 7, 10, 13)
FIT_CHUNK = 4096  # hands per batched Levenberg-Marquardt solve
MESH_CHUNK = 1024  # hands per MANO layer call in mano2mesh
MIRROR = np.array((-1.0, 1.0, 1.0), np.float32)  # the left hand = the right model mirrored in x
STALL_EVERY, STALL_GAIN = 10, 0.001  # the fit stops once no hand's energy fell by more than 0.1 % over the last 10 steps


def _rodrigues(aa):
    """Rotation matrices of axis-angle vectors (Taylor-expanded near zero, differentiable)."""
    theta = aa.norm(dim=-1, keepdim=True)
    small = theta < 0.0001
    safe = theta.clamp_min(1e-12)
    a = torch.where(small, 1 - theta.square() / 6, torch.sin(theta) / safe)
    b = torch.where(small, 0.5 - theta.square() / 24, (1 - torch.cos(theta)) / safe.square())
    x, y, z = aa.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(k)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def _matrices(axis_angles):
    """Rotation matrices (float32 numpy) of axis-angle vectors, via scipy."""
    aa = np.asarray(axis_angles, np.float64)
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*aa.shape[:-1], 3, 3).astype(np.float32)


def _kabsch(source, target):
    """Rotations that best map the centred points `source` onto the centred points `target` (batched SVD)."""
    h = torch.einsum("nki,nkj->nij", source, target)
    u, _, vt = torch.linalg.svd(h)
    sign = torch.det(vt.transpose(-1, -2) @ u.transpose(-1, -2))
    one = torch.ones_like(sign)
    return vt.transpose(-1, -2) @ torch.diag_embed(torch.stack((one, one, sign), -1)) @ u.transpose(-1, -2)


def _bounded_pose(coordinates):
    """Maps unconstrained pose coordinates to 15 axis-angle vectors whose angles stay below pi (a smooth tanh squash)."""
    aa = coordinates.reshape(-1, 15, 3)
    radius = math.pi - 1e-06
    radius -= 8 * torch.finfo(aa.dtype).eps * radius
    norm = aa.norm(dim=-1, keepdim=True)
    factor = radius * torch.tanh(norm / radius) / norm.clamp_min(1e-08)
    return aa * torch.where(norm < 1e-08, torch.ones_like(factor), factor)


def _gauss_jordan(a, rhs):
    """Solves the batched linear systems a x = rhs by Gauss-Jordan elimination without pivoting (a is damped and positive definite)."""
    n = a.shape[-1]
    augmented = torch.cat((a, rhs), -1)
    for k in range(n):
        row = augmented[:, k] / augmented[:, k, k, None]
        augmented = augmented - augmented[:, :, k, None] * row[:, None, :]
        augmented[:, k] = row
    return augmented[:, :, n:]


class ManoModel:
    """The right MANO hand (smplx, no PCA). Joints = the 16 kinematic joints of the rest-shape regressor posed along the chain, then the
    five fingertip vertices; both hands use the right model, the left mirrored in x, plus `cam_trans`."""

    tip_vertices = (744, 320, 443, 554, 671)
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]
    openpose = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)  # the MANO index of each OpenPose joint

    def __init__(self, model_dir, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with contextlib.redirect_stdout(io.StringIO()):  # smplx prints a warning about the 10 shape coefficients MANO has
            self.layer = smplx.MANOLayer(str(model_dir), is_rhand=True, use_pca=False)
        self.layer = self.layer.to(self.device).eval().requires_grad_(False)
        self.faces = self.layer.faces
        self.hand_mean = self.layer.hand_mean.reshape(15, 3)
        tips = list(self.tip_vertices)
        regressor, shapedirs = self.layer.J_regressor, self.layer.shapedirs
        with tf32(False):  # full float32 products, whatever the caller's TF32 setting
            self.joint_template = regressor @ self.layer.v_template
            self.joint_dirs = torch.einsum("ji,ikb->jkb", regressor, shapedirs)
        self.tip_template = self.layer.v_template[tips]
        self.tip_dirs = shapedirs[tips]
        self.tip_posedirs = self.layer.posedirs.reshape(135, 778, 3)[:, tips]
        self.tip_weights = self.layer.lbs_weights[tips]
        self.chain = torch.as_tensor(self.parents[1:16], device=self.device)
        self.identity = torch.eye(3, device=self.device)
        self.homogeneous = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device)

    def joints(self, rotations, betas):
        """The 21 joints [n, 21, 3] of right hands from rotation matrices [n, 16, 3, 3] (global orient, then the hand pose with its mean)
        and shapes [n, 10]: forward kinematics of the rest joints, and the tip vertices skinned in closed form."""
        n = len(rotations)
        rest = self.joint_template + torch.einsum("jkb,nb->njk", self.joint_dirs, betas)
        offsets = torch.cat((rest[:, :1], rest[:, 1:] - rest[:, self.chain]), 1)
        local = torch.cat((torch.cat((rotations, offsets[..., None]), -1), self.homogeneous.expand(n, 16, 1, 4)), -2)
        world = [local[:, 0]]
        for joint in range(1, 16):
            world.append(world[self.parents[joint]] @ local[:, joint])
        world = torch.stack(world, 1)
        rotation, posed = world[:, :, :3, :3], world[:, :, :3, 3]
        skin = posed - (rotation @ rest[..., None])[..., 0]
        feature = (rotations[:, 1:] - self.identity).reshape(n, 135)
        tips = self.tip_template + torch.einsum("jkb,nb->njk", self.tip_dirs, betas) + torch.einsum("np,pjk->njk", feature, self.tip_posedirs)
        tips = (torch.einsum("vj,njab->nvab", self.tip_weights, rotation) @ tips[..., None])[..., 0]
        tips = tips + torch.einsum("vj,nja->nva", self.tip_weights, skin)
        return torch.cat((posed, tips), 1)

    def _hand(self, parameters, rows, side):
        """Rotations, shapes, translations [n, 1, 3] and the mirror sign of one side's rows of a MANO parameter dict, as tensors."""
        tensor = lambda value: torch.as_tensor(np.asarray(value, np.float32), device=self.device)
        rotations = tensor(np.concatenate((parameters["global_orient"][rows, side], parameters["hand_pose"][rows, side]), 1))
        sign = tensor(MIRROR if side == 0 else np.ones(3, np.float32))
        return rotations, tensor(parameters["betas"][rows, side]), tensor(parameters["cam_trans"][rows, side])[:, None], sign

    @torch.no_grad()
    def mano2joint(self, parameters):
        """The 21 camera-space joints [T, 2, 21, 3] of a MANO parameter dict (NaN where a hand is not valid)."""
        result = np.full((len(parameters["cam_trans"]), 2, 21, 3), np.nan, np.float32)
        for side in (0, 1):
            rows = np.flatnonzero(parameters["valid"][:, side])
            if len(rows):
                rotations, betas, translation, sign = self._hand(parameters, rows, side)
                result[rows, side] = (self.joints(rotations, betas) * sign + translation).cpu().numpy()
        return result

    @torch.no_grad()
    def mano2mesh(self, parameters):
        """The 778 camera-space mesh vertices [T, 2, 778, 3] of a MANO parameter dict (NaN where a hand is not valid)."""
        result = np.full((len(parameters["cam_trans"]), 2, 778, 3), np.nan, np.float32)
        for side in (0, 1):
            rows = np.flatnonzero(parameters["valid"][:, side])
            for start in range(0, len(rows), MESH_CHUNK):
                rotations, betas, translation, sign = self._hand(parameters, rows[start : start + MESH_CHUNK], side)
                vertices = self.layer(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:], betas=betas).vertices
                result[rows[start : start + MESH_CHUNK], side] = (vertices * sign + translation).cpu().numpy()
        return result

    def joint2mano(self, joints3d, valid, iterations=60, pose_prior=0.03, joint_noise_mm=1.0):
        """Fits MANO (rotation matrices, a shape per hand and frame, `cam_trans`) to camera-space joints [T, 2, 21, 3] where `valid`; each hand
        is fitted on its own by Levenberg-Marquardt on the centred joints with a pose prior and a shape prior (NaN where not fitted)."""
        joints3d = np.asarray(joints3d, np.float32)
        valid = np.asarray(valid, bool) & np.isfinite(joints3d).all((-1, -2))
        n = len(joints3d)
        result = dict(global_orient=np.full((n, 2, 1, 3, 3), np.nan, np.float32), hand_pose=np.full((n, 2, 15, 3, 3), np.nan, np.float32),
                      betas=np.full((n, 2, 10), np.nan, np.float32), cam_trans=np.full((n, 2, 3), np.nan, np.float32), valid=valid)
        frames, sides = np.nonzero(valid)
        if len(frames):
            mirror = np.where(sides[:, None] == 0, MIRROR, 1.0).astype(np.float32)
            threads = torch.get_num_threads()
            torch.set_num_threads(min(threads, 8))
            try:
                fitted = self._fit(joints3d[frames, sides] * mirror[:, None], iterations, pose_prior, joint_noise_mm)
            finally:
                torch.set_num_threads(threads)
            fitted["cam_trans"] = fitted["cam_trans"] * mirror
            for key, value in fitted.items():
                result[key][frames, sides] = value
        return result

    def _palm_start(self, target):
        """Starting global orientations (axis-angle): the rotation of the mean-shape rest palm onto each target palm."""
        rest = self.joints(torch.eye(3, device=self.device).expand(1, 16, 3, 3), torch.zeros(1, 10, device=self.device))[:, list(PALM)]
        palm = target[:, list(PALM)]
        rotation = _kabsch((rest - rest.mean(1, keepdim=True)).expand(len(target), -1, -1), palm - palm.mean(1, keepdim=True))
        return torch.as_tensor(Rotation.from_matrix(rotation.cpu().numpy()).as_rotvec(), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    @tf32(False)
    def _fit(self, joints, iterations, pose_prior, joint_noise_mm):
        """MANO parameters of right hands [n, 21, 3]; the translation matches the joint centroids."""
        target = torch.as_tensor(joints, device=self.device)
        x = torch.cat((self._palm_start(target), torch.zeros(len(target), 55, device=self.device)), 1)
        self._solve(target, x, iterations, pose_prior, joint_noise_mm)
        betas = x[:, 48:]
        rotations = _matrices(torch.cat((x[:, None, :3], _bounded_pose(x[:, 3:48]) + self.hand_mean), 1).cpu().numpy())
        translation = target.mean(1) - self.joints(torch.as_tensor(rotations, device=self.device), betas).mean(1)
        return dict(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:], betas=betas.cpu().numpy(), cam_trans=translation.cpu().numpy())

    def _solve(self, target, x, iterations, pose_prior, joint_noise_mm):
        """Refines x = [global orient (3), pose coordinates (45), betas (10)] in place: batched Levenberg-Marquardt with forward-difference
        Jacobians on the centred joints in millimetres (CUDA-graph replays on the GPU); stops early once no hand's energy falls by more than 0.1 %
        over 10 steps."""
        device, mean, n, step_size = self.device, self.hand_mean, x.shape[1], 0.001
        scale, pose_weight, shape_weight = 1000.0 / math.sqrt(21), math.sqrt(pose_prior), joint_noise_mm / math.sqrt(21)
        basis, identity = torch.eye(n, device=device) * step_size, torch.eye(n, device=device)

        def solver(target, x):
            m = len(x)
            goal = (target - target.mean(1, keepdim=True)) * scale

            def residuals(x):
                copies = len(x) // m
                offsets, betas = _bounded_pose(x[:, 3:48]), x[:, 48:]
                joints = self.joints(_rodrigues(torch.cat((x[:, None, :3], offsets + mean), 1)), betas)
                data = ((joints - joints.mean(1, keepdim=True)) * scale - goal.repeat(copies, 1, 1)).reshape(len(x), -1)
                return torch.cat((data, pose_weight * offsets.reshape(len(x), -1), shape_weight * betas), 1)

            damping = torch.full((m,), 10.0, device=device)
            r = residuals(x)
            energy = r.square().sum(1)

            def step():
                jacobian = ((residuals((x[None] + basis[:, None]).reshape(n * m, n)).reshape(n, m, -1) - r[None]) / step_size).permute(1, 2, 0)
                normal = jacobian.transpose(1, 2) @ jacobian
                damped = normal + (damping[:, None, None] * normal.diagonal(dim1=1, dim2=2)[..., None] + 1e-09) * identity
                x_new = x - _gauss_jordan(damped, jacobian.transpose(1, 2) @ r[..., None])[..., 0]
                r_new = residuals(x_new)
                e_new = r_new.square().sum(1)
                accept = torch.isfinite(e_new) & (e_new < energy)
                x.copy_(torch.where(accept[:, None], x_new, x))
                r.copy_(torch.where(accept[:, None], r_new, r))
                energy.copy_(torch.where(accept, e_new, energy))
                damping.copy_(torch.where(accept, damping * 0.1, damping * 10.0).clamp(1e-09, 1e09))

            return step, energy

        steps, energies = zip(*(solver(target[a:a + FIT_CHUNK], x[a:a + FIT_CHUNK]) for a in range(0, len(x), FIT_CHUNK)))
        energy = lambda: torch.cat(energies)
        settled = energy()
        with tqdm(total=iterations, desc="ViDiHand2: fitting MANO to the predicted joints", unit="step") as bar:
            runs, done = steps, 0
            if device.type == "cuda" and iterations > 1:
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for step in steps:
                        step()
                torch.cuda.current_stream(device).wait_stream(stream)
                graphs, pool = [torch.cuda.CUDAGraph() for _ in steps], torch.cuda.graph_pool_handle()
                for graph, step in zip(graphs, steps):
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
                        step()
                runs, done = [graph.replay for graph in graphs], 1
                bar.update(done)
            for step_index in range(done, iterations):
                for run in runs:
                    run()
                bar.update()
                if step_index % STALL_EVERY == STALL_EVERY - 1:
                    if bool((settled - energy() <= STALL_GAIN * settled).all()):
                        bar.update(iterations - step_index - 1)
                        break
                    settled = energy()
