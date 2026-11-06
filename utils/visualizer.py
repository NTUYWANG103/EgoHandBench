"""Overlay MANO meshes or joints on calibrated RGB frames."""

from pathlib import Path

import numpy as np
import torch
import cv2

from utils.io import load_video
from utils.mano import ManoModel


class Visualizer:
    """Render meshes on CPU/GPU and joint skeletons in fixed hand colours."""

    def __init__(self, backend="auto"):
        if backend not in ("auto", "cpu", "gpu"):
            raise ValueError("backend must be auto, cpu or gpu")
        self._model = None
        self._context = None
        self._dr = None
        self._gpu_required = backend == "gpu"
        if backend != "cpu" and torch.cuda.is_available():
            try:
                import nvdiffrast.torch as dr

                self._dr = dr
            except (ImportError, OSError):
                pass
        if self._gpu_required and self._dr is None:
            raise RuntimeError("GPU rendering requires CUDA and nvdiffrast")
        self.backend = "gpu" if self._dr is not None else "cpu"

    @staticmethod
    def _video(prediction, inputs):
        video = inputs.get("image", inputs.get("video", inputs.get("video_path")))
        video = (
            load_video(video) if isinstance(video, (str, Path)) else np.asarray(video)
        )
        if video.ndim == 3:
            video = video[None]
        camera = inputs["camera"]
        if len(video) != len(camera) or not np.array_equal(
            prediction["source_frame_ids"], camera.source_frame_ids
        ):
            raise ValueError("RGB, prediction and camera timelines must match")
        return video

    def render_mano(self, prediction, inputs):
        """Return RGB frames with camera-space MANO surfaces overlaid."""
        video = self._video(prediction, inputs)
        camera = inputs["camera"]
        if self._model is None:
            self._model = ManoModel()
        geometry = dict(
            prediction, cam_trans=np.asarray(prediction["cam_trans"], np.float32)
        )
        vertices = self._model.mano2mesh(geometry)
        output = video.copy()
        for index, image in enumerate(video):
            meshes = []
            for side in (0, 1):
                if not prediction["valid"][index, side]:
                    continue
                mesh = np.asarray(vertices[index, side], np.float64)
                faces = self._model.faces
                if side == 0:
                    faces = faces[:, [0, 2, 1]]
                colours = self._colours(mesh, faces, side)
                visible = (mesh[faces, 2] > 1e-7).all(-1)
                meshes.append((mesh, faces[visible], colours))
            if not meshes:
                continue
            if self.backend == "gpu":
                try:
                    output[index] = self._render_gpu(image, meshes, camera.K[index])
                    continue
                except (ImportError, OSError, RuntimeError):
                    if self._gpu_required:
                        raise
                    self.backend = "cpu"
            output[index] = self._render_cpu(image, meshes, camera.K[index])
        return output

    def render_joint(self, prediction, inputs):
        """Overlay joints in fixed per-finger colours."""
        output = self._video(prediction, inputs).copy()
        camera = inputs["camera"]
        parents = (0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9)
        colours = (
            (
                (176, 196, 222),  # wrist
                (0, 95, 158), (0, 130, 216), (72, 182, 255),  # index
                (0, 26, 158), (0, 36, 216), (72, 102, 255),  # middle
                (105, 0, 158), (144, 0, 216), (194, 72, 255),  # pinky
                (37, 0, 158), (50, 0, 216), (115, 72, 255),  # ring
                (0, 158, 158), (0, 216, 216), (72, 255, 255),  # thumb
                # tips: thumb, index, middle, ring, pinky
                (153, 255, 255), (153, 214, 255), (153, 170, 255),
                (177, 153, 255), (221, 153, 255),
            ),
            (
                (255, 255, 255),  # wrist
                (158, 79, 0), (216, 108, 0), (255, 164, 72),  # index
                (158, 158, 0), (216, 216, 0), (255, 255, 72),  # middle
                (0, 158, 53), (0, 216, 72), (72, 255, 133),  # pinky
                (53, 158, 0), (72, 216, 0), (133, 255, 72),  # ring
                (158, 0, 0), (216, 0, 0), (255, 72, 72),  # thumb
                # tips: thumb, index, middle, ring, pinky
                (255, 153, 153), (255, 204, 153), (255, 255, 153),
                (187, 255, 153), (153, 255, 187),
            ),
        )
        for index, image in enumerate(output):
            pixels = np.zeros((2, 21, 2), np.int32)
            visible = np.zeros((2, 21), bool)
            for side in (0, 1):
                if not prediction["valid"][index, side]:
                    continue
                joints = np.asarray(prediction["joints3d"][index, side])
                good = np.isfinite(joints).all(-1) & (joints[:, 2] > 1e-7)
                projected = joints[good] @ camera.K[index].T
                pixels[side, good] = np.rint(
                    projected[:, :2] / projected[:, 2:]
                ).astype(np.int32)
                visible[side] = good
            # Draw all bones first, then all joints, left hand before right.
            for side in (0, 1):
                for child, parent in enumerate(parents, 1):
                    if not (visible[side, child] and visible[side, parent]):
                        continue
                    a, b = tuple(pixels[side, parent]), tuple(pixels[side, child])
                    colour = tuple(
                        (x + y) // 2
                        for x, y in zip(colours[side][child], colours[side][parent])
                    )
                    cv2.line(image, a, b, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.line(image, a, b, colour, 1, cv2.LINE_AA)
            for side in (0, 1):
                for joint in np.flatnonzero(visible[side]):
                    point = tuple(pixels[side, joint])
                    radius = 3 if joint == 0 else 2
                    cv2.circle(image, point, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(image, point, radius, colours[side][joint], -1, cv2.LINE_AA)
        return output

    @staticmethod
    def _colours(vertices, faces, side):
        triangles = vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        colours = 0.5 + 0.5 * normals
        if side == 0:
            colours[:, 2] = 1.0 - colours[:, 2]
        tint = np.array((0.25, 0.60, 1.00) if side == 0 else (1.00, 0.55, 0.20))
        return np.clip(colours * 0.60 + tint * 0.40, 0.0, 1.0)

    @staticmethod
    def _render_cpu(image, meshes, intrinsics):
        """Perspective-correct triangles sharing one depth buffer."""
        height, width = image.shape[:2]
        output = image.copy()
        depth = np.full((height, width), np.inf)
        for vertices, faces, colours in meshes:
            projected = vertices @ intrinsics.T
            pixels = projected[:, :2] / projected[:, 2:]
            for face in faces:
                points = pixels[face]
                (x0, y0), (x1, y1), (x2, y2) = points
                denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denominator) < 1e-10:
                    continue
                left, top = np.maximum(0, np.floor(points.min(0))).astype(int)
                right, bottom = np.minimum(
                    [width - 1, height - 1], np.ceil(points.max(0))
                ).astype(int)
                if left > right or top > bottom:
                    continue
                y, x = np.mgrid[top : bottom + 1, left : right + 1]
                first = (
                    (y1 - y2) * (x + 0.5 - x2) + (x2 - x1) * (y + 0.5 - y2)
                ) / denominator
                second = (
                    (y2 - y0) * (x + 0.5 - x2) + (x0 - x2) * (y + 0.5 - y2)
                ) / denominator
                barycentric = np.stack((first, second, 1 - first - second), -1)
                weighted = barycentric / vertices[face, 2]
                z = 1.0 / weighted.sum(-1)
                region = depth[top : bottom + 1, left : right + 1]
                update = (barycentric >= 0).all(-1) & (z < region)
                if not update.any():
                    continue
                colour = (weighted @ colours[face]) * z[..., None]
                region[update] = z[update]
                output[top : bottom + 1, left : right + 1][update] = np.rint(
                    np.clip(colour[update], 0, 1) * 255
                ).astype(np.uint8)
        return output

    @torch.no_grad()
    def _render_gpu(self, image, meshes, intrinsics):
        """Rasterize both hands together; the nearest surface wins."""
        if self._context is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
            self._context = self._dr.RasterizeCudaContext(device=self._device)
        height, width = image.shape[:2]
        positions, triangles, attributes = [], [], []
        offset = 0
        positive_depth = np.concatenate(
            [mesh[0][mesh[1], 2].ravel() for mesh in meshes]
        )
        if not len(positive_depth):
            return image.copy()
        near, far = max(1e-7, positive_depth.min() * 0.5), positive_depth.max() * 2.0
        for vertices, faces, colours in meshes:
            z = vertices[:, 2]
            projected = vertices @ intrinsics.T
            # Tensor row 0 is image row 0: positive camera Y maps downwards.
            clip = np.stack(
                (
                    2 * projected[:, 0] / width - z,
                    2 * projected[:, 1] / height - z,
                    (far + near) / (far - near) * z - 2 * far * near / (far - near),
                    z,
                ),
                axis=-1,
            )
            positions.append(clip)
            triangles.append(faces + offset)
            attributes.append(colours)
            offset += len(vertices)
        positions = torch.as_tensor(
            np.concatenate(positions), dtype=torch.float32, device=self._device
        )[None]
        triangles = torch.as_tensor(
            np.concatenate(triangles), dtype=torch.int32, device=self._device
        )
        attributes = torch.as_tensor(
            np.concatenate(attributes), dtype=torch.float32, device=self._device
        )[None]
        raster, _ = self._dr.rasterize(
            self._context, positions, triangles, resolution=[height, width]
        )
        colour, _ = self._dr.interpolate(attributes, raster, triangles)
        mask = (raster[0, ..., 3] > 0).cpu().numpy()
        rendered = (colour[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        output = image.copy()
        output[mask] = rendered[mask]
        return output


def main():
    """Render saved predictions for a prepared local dataset."""
    import argparse
    from tqdm import tqdm
    from utils.io import load_npz, load_sample, manifest_file, manifest_records, save_video

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mano", type=Path, help="Prediction MANO directory")
    source.add_argument("--joint", type=Path, help="Joint prediction directory")
    parser.add_argument(
        "--camera", type=Path, required=True, help="Camera JSON directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Rendered MP4 directory"
    )
    args = parser.parse_args()
    predictions = args.mano if args.mano is not None else args.joint
    manifest, records = manifest_records(args.camera.parent / "manifests/split.json")
    visualizer = Visualizer()
    paths = (
        [
            predictions / (name + ".npz")
            for name in manifest["test"]
            if (predictions / (name + ".npz")).is_file()
        ]
        if manifest is not None
        else sorted(predictions.glob("*.npz"))
    )
    render = visualizer.render_mano if args.mano is not None else visualizer.render_joint
    for path in tqdm(paths, desc="Rendering", unit="clip"):
        names = {"video": path.stem + ".mp4", "camera": path.stem + ".json"}
        for kind in names:
            value = manifest_file(records, path.stem, kind)
            if value:
                names[kind] = Path(value).name
        sample = load_sample(args.video / names["video"], args.camera / names["camera"])
        video = render(load_npz(path), sample)
        save_video(args.output / (path.stem + ".mp4"), video, sample["camera"].fps)


if __name__ == "__main__":
    main()
