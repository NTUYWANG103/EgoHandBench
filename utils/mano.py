"""RIGHT MANO geometry (left hand mirrored before translation), mesh fitting and camera refit."""

from contextlib import nullcontext
import inspect

import numpy as np
import torch

PARAM_KEYS = ("global_orient", "hand_pose", "betas", "cam_trans")
# Inverse of the native MANO-to-OpenPose map; preserve native tip locations.
OPENPOSE_TO_MANO = (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20)


def enable_mano_pickle_compatibility():
    """Restore the NumPy and inspect names that the licensed MANO pickles (via chumpy) expect."""
    for name, value in dict(
        bool=bool,
        int=int,
        float=float,
        complex=complex,
        object=object,
        unicode=str,
        str=str,
    ).items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def axis_angle_to_rotmat(value):
    from smplx.lbs import batch_rodrigues

    return batch_rodrigues(value.reshape(-1, 3)).reshape(*value.shape[:-1], 3, 3)


class ManoModel:
    """Convert MANO dictionaries, meshes and MANO16-plus-five-tip joints."""

    tip_vertices = (744, 320, 443, 554, 671)

    def __init__(self, model_dir=None, device="cpu"):
        if model_dir is None:
            from .download_data import canonical_mano_dir

            model_dir = canonical_mano_dir()
        enable_mano_pickle_compatibility()
        import smplx

        self.device = device
        if torch.device(device).type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.layer = (
            smplx.MANOLayer(str(model_dir), is_rhand=True, num_pca_comps=45)
            .to(device)
            .eval()
        )
        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)
        self.faces = self.layer.faces

    def _joints(self, output):
        return torch.cat(
            (output.joints[:, :16], output.vertices[:, self.tip_vertices]), dim=1
        )

    def forward(self, rotations, betas, translation, side):
        """Decode one hand batch; side 0 is left and side 1 is right."""
        precise = np.asarray(translation).dtype == np.dtype(np.float64)
        precision = (
            torch.autocast(device_type=str(self.device).split(":")[0], enabled=False)
            if precise
            else nullcontext()
        )
        with torch.no_grad(), precision:

            def tensor(value):
                return torch.as_tensor(value, dtype=torch.float32, device=self.device)

            rotations = tensor(rotations)
            output = self.layer(
                global_orient=rotations[:, :1],
                hand_pose=rotations[:, 1:],
                betas=tensor(betas),
            )
            sign = tensor([-1 if side == 0 else 1, 1, 1])
            dtype = torch.float64 if precise else torch.float32
            translation = torch.as_tensor(translation, dtype=dtype, device=self.device)[
                :, None
            ]
            vertices = (output.vertices * sign).to(dtype) + translation
            joints = (self._joints(output) * sign).to(dtype) + translation
        return vertices.cpu().numpy(), joints.cpu().numpy()

    def _convert(self, parameters, mesh):
        count = len(parameters["cam_trans"])
        dtype = (
            np.float64 if parameters["cam_trans"].dtype == np.float64 else np.float32
        )
        result = np.full((count, 2, 778 if mesh else 21, 3), np.nan, dtype)
        rotations = np.concatenate(
            [parameters["global_orient"], parameters["hand_pose"]], axis=2
        )
        for side in (0, 1):
            rows = np.flatnonzero(parameters["valid"][:, side])
            for start in range(0, len(rows), 256):
                index = rows[start : start + 256]
                vertices, joints = self.forward(
                    rotations[index, side],
                    parameters["betas"][index, side],
                    parameters["cam_trans"][index, side],
                    side,
                )
                result[index, side] = vertices if mesh else joints
        return result

    def mano2joint(self, parameters):
        """Return camera-space joints [T, 2, 21, 3], in metres."""
        return self._convert(parameters, mesh=False)

    def mano2mesh(self, parameters):
        """Return camera-space vertices [T, 2, 778, 3], in metres."""
        return self._convert(parameters, mesh=True)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _mesh_fit(self, rotations, betas, vertices, side, batch=128):
        from scipy.spatial.transform import Rotation

        frame_count = len(vertices)
        reflection = np.diag([-1.0, 1.0, 1.0]) if side == 0 else np.eye(3)
        initial = (
            (reflection @ rotations @ reflection).astype(np.float32)
            if side == 0
            else rotations.copy()
        )
        initial_axis_angles = self._tensor(
            Rotation.from_matrix(initial.reshape(-1, 3, 3).astype(np.float64))
            .as_rotvec()
            .reshape(frame_count, 48),
        )
        axis_angles = torch.nn.Parameter(initial_axis_angles.clone())
        shape_center = np.median(betas, axis=0)
        shape_limit = np.maximum(3.0, np.max(np.abs(betas), axis=0) + 0.1)
        shape = torch.nn.Parameter(self._tensor(shape_center))
        side_sign = self._tensor([-1 if side == 0 else 1, 1, 1])
        shape_bound = self._tensor(shape_limit)
        target_vertices_mm = self._tensor(vertices) * 1000
        optimizer = torch.optim.LBFGS(
            [axis_angles, shape],
            max_iter=200,
            max_eval=260,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-06,
            tolerance_change=1e-09,
        )

        def closure():
            optimizer.zero_grad()
            total = 0.0
            for start in range(0, frame_count, batch):
                stop = min(frame_count, start + batch)
                pose_rotations = axis_angle_to_rotmat(axis_angles[start:stop].reshape(-1, 16, 3))
                fitted_betas = shape.clamp(-shape_bound, shape_bound).expand(
                    stop - start, -1
                )
                posed_vertices_mm = (
                    self.layer(
                        global_orient=pose_rotations[:, :1],
                        hand_pose=pose_rotations[:, 1:],
                        betas=fitted_betas,
                    ).vertices
                    * side_sign
                    * 1000
                )
                residual = posed_vertices_mm - target_vertices_mm[start:stop]
                residual = residual - residual.mean(1, keepdim=True)
                loss = (
                    residual.square().mean()
                    + 0.001
                    * (axis_angles[start:stop] - initial_axis_angles[start:stop])
                    .square()
                    .mean()
                ) * ((stop - start) / frame_count)
                loss.backward()
                total += float(loss.detach())
            loss = 0.001 * (shape - self._tensor(shape_center)).square().mean()
            loss.backward()
            total += float(loss.detach())
            return axis_angles.new_tensor(total)

        optimizer.step(closure)
        with torch.no_grad():
            fitted_rotations = axis_angle_to_rotmat(axis_angles.reshape(-1, 16, 3)).cpu().numpy()
            fitted_betas = (
                shape.clamp(-shape_bound, shape_bound)
                .expand(frame_count, -1)
                .cpu()
                .numpy()
                .copy()
            )
            translations = []
            for start in range(0, frame_count, batch):
                pose_rotations = self._tensor(fitted_rotations[start : start + batch])
                posed_vertices_mm = (
                    self.layer(
                        global_orient=pose_rotations[:, :1],
                        hand_pose=pose_rotations[:, 1:],
                        betas=self._tensor(fitted_betas[start : start + batch]),
                    ).vertices
                    * side_sign
                    * 1000
                )
                translations.append(
                    (
                        (
                            target_vertices_mm[start : start + batch] - posed_vertices_mm
                        ).mean(1)
                        / 1000
                    )
                    .cpu()
                    .numpy()
                )
        return (fitted_rotations, fitted_betas, np.concatenate(translations))

    def mesh2mano(self, rotations, betas, vertices, side):
        """Represent native per-frame meshes [N, 778, 3] of one side in this MANO.

        Rotations [N, 16, 3, 3] and betas [N, 10] are the native ones; frames whose
        direct decode is off by 0.011 mm or more are refit, keeping the better fit.
        Returns rotations, betas and translations for side 0 (left) or 1 (right).
        """
        native_rotations = np.asarray(rotations, dtype=np.float32)
        native_betas = np.asarray(betas, dtype=np.float32)
        native_vertices = np.asarray(vertices, dtype=np.float32)
        frame_count = len(native_vertices)
        reflection = np.diag([-1.0, 1.0, 1.0])
        canonical_rotations = (
            (reflection @ native_rotations @ reflection).astype(np.float32)
            if side == 0
            else native_rotations.copy()
        )
        canonical_betas = native_betas.copy()
        errors, translations = ([], [])
        for start in range(0, frame_count, 256):
            stop = min(frame_count, start + 256)
            zero, _ = self.forward(
                canonical_rotations[start:stop],
                native_betas[start:stop],
                np.zeros((stop - start, 3), np.float32),
                side,
            )
            trans = (native_vertices[start:stop] - zero).mean(1)
            errors.append(
                np.linalg.norm(zero + trans[:, None] - native_vertices[start:stop], axis=-1)
                * 1000
            )
            translations.append(trans)
        initial = np.concatenate(errors)
        translations = np.concatenate(translations)
        direct = initial.max(1) < 0.011
        bad = np.flatnonzero(~direct)
        if len(bad):
            fitted_rotations, fitted_betas, fitted_translations = self._mesh_fit(
                native_rotations[bad], native_betas[bad], native_vertices[bad], side
            )
            # Fitting is approximate: retain an already better initial representation.
            accepted = np.zeros(len(bad), dtype=bool)
            for start in range(0, len(bad), 256):
                stop = min(start + 256, len(bad))
                indices = bad[start:stop]
                initial_vertices, _ = self.forward(
                    canonical_rotations[indices], canonical_betas[indices],
                    translations[indices], side,
                )
                fitted_vertices, _ = self.forward(
                    fitted_rotations[start:stop], fitted_betas[start:stop],
                    fitted_translations[start:stop], side,
                )
                initial_residual = (initial_vertices - native_vertices[indices]).astype(np.float64)
                fitted_residual = (fitted_vertices - native_vertices[indices]).astype(np.float64)
                initial_centered = initial_residual - initial_residual.mean(1, keepdims=True)
                fitted_centered = fitted_residual - fitted_residual.mean(1, keepdims=True)
                initial_mse = np.square(initial_centered).mean((1, 2))
                fitted_mse = np.square(fitted_centered).mean((1, 2))
                initial_max = np.linalg.norm(initial_residual, axis=-1).max(1)
                fitted_max = np.linalg.norm(fitted_residual, axis=-1).max(1)
                accepted[start:stop] = (
                    np.isfinite(fitted_residual).all((1, 2))
                    & (fitted_mse <= initial_mse) & (fitted_max <= initial_max)
                )
            indices = bad[accepted]
            canonical_rotations[indices] = fitted_rotations[accepted]
            canonical_betas[indices] = fitted_betas[accepted]
            translations[indices] = fitted_translations[accepted]
        return (canonical_rotations, canonical_betas, translations)


def refit_translation(relative, uv, intrinsics, prior):
    """Translate root-relative joints onto their pixel rays under intrinsics (a frozen point-to-ray solve).

    No pose/shape optimisation and no GT joints; rows the solve cannot trust keep the prior translation.
    """
    with torch.autocast("cpu", enabled=False):
        rel = torch.as_tensor(relative, dtype=torch.float32)
        points = torch.as_tensor(uv, dtype=torch.float32)
        initial = torch.as_tensor(prior, dtype=torch.float32)
        valid = torch.isfinite(rel).all(-1) & torch.isfinite(points).all(-1)
        weights = valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        rel = torch.where(valid[..., None], rel, 0.0)
        points = torch.where(valid[..., None], points, 0.0)
        k = np.asarray(intrinsics)
        if k[0, 1] == 0 and k[1, 0] == 0 and np.array_equal(k[2], [0, 0, 1]):
            rays = torch.stack(((points[..., 0] - k[0, 2]) / k[0, 0],
                                (points[..., 1] - k[1, 2]) / k[1, 1],
                                torch.ones_like(points[..., 0])), -1)
        else:
            homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
            rays = torch.linalg.solve(torch.as_tensor(k, dtype=torch.float32),
                                      homogeneous.transpose(-1, -2)).transpose(-1, -2)
        rays = torch.nn.functional.normalize(rays, dim=-1)
        identity = torch.eye(3)
        projector = identity - rays[..., :, None] * rays[..., None, :]
        information = (weights[..., None, None] * projector).sum(-3)
        gradient = (weights[..., None] * (projector @ rel[..., None]).squeeze(-1)).sum(-2)
        eigenvalues = torch.linalg.eigvalsh(information)
        condition = torch.where(
            eigenvalues[..., 0] > 0,
            eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-12),
            float("inf"),
        )
        solution, info = torch.linalg.solve_ex(
            information + 0.0 * identity, -gradient + 0.0 * initial
        )
        depth = torch.as_tensor(relative, dtype=torch.float32)[..., 2] + solution[..., None, 2]
        usable = (
            (valid.sum(-1) >= 2) & (condition <= 1e6) & (info == 0)
            & torch.isfinite(solution).all(-1)
            & ((depth > 1e-3) | ~torch.isfinite(torch.as_tensor(relative)).all(-1)).all(-1)
        )
        return torch.where(usable[..., None], solution, initial).numpy()


def source_prediction(raw, camera, native_intrinsics=None):
    """Return native MANO21 joints moved into the supplied camera, and that per-hand shift.

    native_intrinsics: per-hand native [fx, fy, cx, cy], default raw["projection_intrinsics"].
    No canonical fitting is involved, so these joints are identical with or without the MANO export.
    """
    valid = np.asarray(raw["presence"], dtype=bool)
    joints = np.asarray(raw["joints_cam"], dtype=np.float64)[:, :, OPENPOSE_TO_MANO].copy()
    joints[~valid] = np.nan
    source = raw["projection_intrinsics"] if native_intrinsics is None else native_intrinsics
    target = np.asarray(camera.K, dtype=np.float64)
    target_parameters = target[:, (0, 1, 0, 1), (0, 1, 2, 2)]
    native = np.broadcast_to(source, (*valid.shape, 4))[valid]
    supplied = np.broadcast_to(target_parameters[:, None, :], (*valid.shape, 4))[valid]
    supplied_k = np.broadcast_to(target[:, None], (*valid.shape, 3, 3))[valid]
    points = joints[valid]
    if not np.isfinite(target).all() or np.any(target_parameters[:, :2] <= 0):
        raise ValueError("Finite intrinsics with positive focal lengths required")
    if not np.isfinite(native).all() or np.any(native[:, :2] <= 0):
        raise ValueError("Invalid native camera")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite native joints on a claimed hand")
    delta = np.zeros((*valid.shape, 3), dtype=np.float64)
    same = (np.all(np.isclose(native, supplied, rtol=1e-6, atol=1e-6), axis=-1)
            & (supplied_k[:, 0, 1] == 0) & (supplied_k[:, 1, 0] == 0)
            & np.all(supplied_k[:, 2] == [0, 0, 1], axis=-1))
    if same.all():
        return dict(joints3d=joints.astype(np.float32), valid=valid.copy(),
                    source_frame_ids=camera.source_frame_ids.copy()), delta
    uv = points[..., :2] / points[..., 2:3] * native[:, None, :2] + native[:, None, 2:]
    depth = np.sqrt(supplied[:, 0] * supplied[:, 1] / (native[:, 0] * native[:, 1])) * points[:, 0, 2]
    if not np.isfinite(uv).all() or np.any(depth <= 0):
        raise ValueError("Invalid native joint projection or wrist depth")
    wrist_pixels = np.column_stack((uv[:, 0], np.ones(len(uv))))
    wrist_rays = np.linalg.solve(supplied_k, wrist_pixels[..., None])[..., 0]
    prior = wrist_rays / wrist_rays[:, 2:3] * depth[:, None]
    shifts = np.zeros((len(points), 3), dtype=np.float64)
    for intrinsics in np.unique(supplied_k, axis=0):
        rows = np.all(supplied_k == intrinsics, axis=(-1, -2))
        shifts[rows] = refit_translation(
            points[rows] - points[rows, :1], uv[rows], intrinsics, prior[rows]
        ) - points[rows, 0]
    shifts[same] = 0.0
    delta[valid] = shifts
    joints[valid] += shifts[:, None]
    return dict(joints3d=joints.astype(np.float32), valid=valid.copy(),
                source_frame_ids=camera.source_frame_ids.copy()), delta


def apply_camera(parameters, delta):
    """Move fitted MANO by the source-derived shift, preserving its fit."""
    translation = np.array(parameters["cam_trans"], dtype=np.float64, copy=True)
    valid = np.asarray(parameters["valid"], dtype=bool)
    translation[valid] += delta[valid]
    return dict(parameters, cam_trans=translation)


def native_to_mano(raw, rotations, source_frame_ids, model, *, reflect_left=True, zero_betas=False):
    """Fit RIGHT MANO (left geometry reflected) to the claimed native meshes raw["vertices_cam"].

    rotations [T, 2, 16, 3, 3] and raw["betas"] (zeros with zero_betas) start model.mesh2mano;
    reflect_left mirrors the native left-hand rotations first, for methods that predict them unmirrored.
    """
    claims = np.asarray(raw["presence"])
    frame_count = len(source_frame_ids)
    rotations = np.asarray(rotations, dtype=np.float32)
    betas = np.asarray(np.zeros((frame_count, 2, 10)) if zero_betas else raw["betas"], dtype=np.float32)
    vertices = np.asarray(raw["vertices_cam"])
    arrays = {
        key: np.full((frame_count, 2, *shape), np.nan, np.float32)
        for key, shape in zip(
            PARAM_KEYS, ((1, 3, 3), (15, 3, 3), (10,), (3,)), strict=True
        )
    }
    reflection = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    for side in (0, 1):
        rows = np.flatnonzero(claims[:, side])
        if not len(rows):
            continue
        initial = rotations[rows, side].copy()
        if reflect_left and side == 0:
            initial = reflection @ initial @ reflection
        fitted_rotations, fitted_betas, translations = model.mesh2mano(
            initial, betas[rows, side].copy(), vertices[rows, side], side
        )
        arrays["global_orient"][rows, side] = fitted_rotations[:, :1]
        arrays["hand_pose"][rows, side] = fitted_rotations[:, 1:]
        arrays["betas"][rows, side] = fitted_betas
        arrays["cam_trans"][rows, side] = translations
    return dict(
        arrays,
        valid=claims.copy(),
        source_frame_ids=np.asarray(source_frame_ids).copy(),
    )


def refit_prediction(raw, camera, model=None, *, joint_only=False, native_intrinsics=None, reflect_left=True):
    """Joints refit to the supplied K, and the MANO export moved by the same shift.

    For networks that predict under their own camera (WiLoR, AnyHand, EgoHandICL, HaMeR, Hamba;
    MINT with its predicted K as native_intrinsics and reflect_left=False).
    """
    joint, delta = source_prediction(raw, camera, native_intrinsics)
    if joint_only:
        return {"mano": None, "joint": joint}
    rotations = np.concatenate([raw["global_orient"], raw["hand_pose"]], axis=2).copy()
    torch.set_num_threads(2)
    parameters = native_to_mano(
        raw, rotations.astype(np.float32), camera.source_frame_ids, model, reflect_left=reflect_left
    )
    return {"mano": apply_camera(parameters, delta), "joint": joint}


def mano_joint_prediction(parameters, model):
    """Joints decoded from the exported MANO on a CPU ManoModel and stored as float32: the benchmark joints of
    HandDGP, HandFlow, InfiniHand, InterWild, OmniHands, PAD-Hand and WildHands."""
    return {
        "joints3d": model.mano2joint(parameters).astype(np.float32),
        "valid": np.asarray(parameters["valid"], bool).copy(),
        "source_frame_ids": np.asarray(parameters["source_frame_ids"]).copy(),
    }


def native_source_prediction(raw, camera):
    """Native OpenPose-order joints as MANO16 + tips, unshifted: for networks that already ran under K."""
    return {
        "joints3d": np.asarray(raw["joints_cam"])[:, :, OPENPOSE_TO_MANO].copy(),
        "valid": np.asarray(raw["presence"], bool).copy(),
        "source_frame_ids": np.asarray(camera.source_frame_ids).copy(),
    }
