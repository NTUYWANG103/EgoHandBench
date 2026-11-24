"""Local files and RGB media, checkpoint paths, cameras and dataset inputs."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_image(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image[..., ::-1].copy()


class VideoReader:
    def __init__(self, source, color="rgb"):
        self.color = color
        self.array = isinstance(source, np.ndarray)
        if self.array:
            self.stream = iter(source)
        else:
            self.stream = imageio_ffmpeg.read_frames(
                str(source),
                pix_fmt="bgr24",
                input_params=["-threads", "1"],
                output_params=[
                    "-map",
                    "0:v:0",
                    "-an",
                    "-threads",
                    "1",
                    "-fps_mode",
                    "passthrough",
                ],
            )
            self.width, self.height = next(self.stream)["size"]

    def read(self):
        try:
            frame = next(self.stream)
        except StopIteration:
            return False, None
        if not self.array:
            frame = np.frombuffer(frame, np.uint8).reshape(self.height, self.width, 3)
        rgb = frame if self.array else frame[..., ::-1]
        return True, (rgb if self.color == "rgb" else rgb[..., ::-1]).copy()

    def __iter__(self):
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def release(self):
        if not self.array:
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def load_video(path):
    with VideoReader(path) as reader:
        frames = list(reader)
    return np.stack(frames)


def save_video(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    stream = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=fps,
        quality=None,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
        # Library formats fps to two decimals; preserve the original exact input rate.
        input_params=["-r", str(fps)],
        output_params=["-threads", "1", "-crf", "18", "-movflags", "+faststart"],
    )
    stream.send(None)
    try:
        for frame in frames:
            stream.send(np.ascontiguousarray(frame[..., ::-1]))
    finally:
        stream.close()


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(checkpoints, section, key):
    root = Path(checkpoints).expanduser().resolve()
    layout = read_json(root / "layout.json")
    configuration = read_json(REPO_ROOT / "baselines" / layout["baseline"] / "config.json")
    return root / configuration["versions"][layout["version"]][section][key]


def weight_path(checkpoints, method):
    return _resolve(checkpoints, "weights", method)


def resource_path(checkpoints, key):
    """Resolve a resource within its downloaded method/version group."""
    return _resolve(checkpoints, "resources", key)


@dataclass
class Camera:
    K: np.ndarray
    canvas: np.ndarray
    E: np.ndarray | None = None
    source_frame_ids: np.ndarray | None = None
    fps: float = 30.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        if self.K.ndim == 2:
            self.K = self.K[None]
        n = len(self.K)
        self.canvas = np.asarray(self.canvas, dtype=np.int64)
        if self.canvas.shape == (2,):
            self.canvas = np.tile(self.canvas, (n, 1))
        if self.E is not None:
            self.E = np.asarray(self.E, dtype=np.float64)
        if self.source_frame_ids is None:
            self.source_frame_ids = np.arange(n, dtype=np.int64)
        self.source_frame_ids = np.asarray(self.source_frame_ids)

    def __len__(self):
        return len(self.K)

    def slice(self, rows):
        ix = np.atleast_1d(np.arange(len(self))[rows])
        metadata = dict(self.metadata)
        for key in ("camera_pose_valid", "camera_motion_valid"):
            if key in metadata:
                values = np.asarray(metadata[key], dtype=bool)[ix].copy()
                if key == "camera_motion_valid":
                    values[:1] = False
                    values[1:] &= np.diff(ix) == 1
                metadata[key] = values.tolist()
        return Camera(
            self.K[ix],
            self.canvas[ix],
            None if self.E is None else self.E[ix],
            self.source_frame_ids[ix],
            self.fps,
            metadata,
        )

    @classmethod
    def load(cls, path):
        data = read_json(path)
        frames = data["frames"]
        matrices = []
        for frame in frames:
            value = frame["intrinsics"]
            if isinstance(value, dict):
                value = [
                    [value["fx"], 0, value["cx"]],
                    [0, value["fy"], value["cy"]],
                    [0, 0, 1],
                ]
            matrices.append(value)
        extrinsics = [frame.get("extrinsics") for frame in frames]
        return cls(
            np.array(matrices),
            np.array([[data["image_width"], data["image_height"]]] * len(frames)),
            (
                np.array(extrinsics)
                if all(x is not None for x in extrinsics)
                else None
            ),
            np.array([frame["source_frame_id"] for frame in frames]),
            data.get("fps", 30.0),
            {k: v for k, v in data.items() if k != "frames"},
        )


def load_sample(path, camera):
    """Load a local RGB image or video with its calibrated camera timeline."""
    path = Path(path).expanduser().resolve()
    camera_path = None
    if not isinstance(camera, Camera):
        camera_path = Path(camera).expanduser().resolve()
        camera = Camera.load(camera_path)
    is_image = path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    video = load_image(path)[None] if is_image else load_video(path)
    sample = dict(camera=camera, name=path.stem)
    sample["image" if is_image else "video"] = video[0] if is_image else video
    if not is_image:
        sample["video_path"] = str(path)
    if camera_path is not None:
        sample["camera_path"] = str(camera_path)
    return sample


class InferenceInput:
    """Adapt local RGB and camera inputs to a method's native record."""

    def __init__(
        self,
        sample,
        *,
        mode,
        requires_video_path=False,
    ):
        self.sample = sample
        self.mode = mode
        self.requires_video_path = requires_video_path

    def __enter__(self):
        inputs = self.sample
        if "image" in inputs:
            if "video" in inputs:
                raise ValueError("Provide image or video, not both")
            image = np.asarray(inputs["image"])
            inputs = dict(inputs, video=image[None])
            inputs.pop("image")
        return self._prepare_record(inputs)

    def __exit__(self, *_):
        pass

    def _prepare_record(self, inputs):
        """Build the native adapter record from validated local input."""
        camera = inputs.get("camera")
        self._validate_camera(camera, inputs)
        # Rebuilding from K, canvas and timeline alone drops E and all metadata:
        # this is the guarantee that no extrinsics ever reach a method.
        camera = Camera(
            camera.K,
            camera.canvas,
            source_frame_ids=camera.source_frame_ids,
            fps=camera.fps,
        )
        source, video = self._prepare_video(inputs, camera)
        width, height = map(int, camera.canvas[0])
        name = inputs.get("name", "clip")
        record = dict(
            clip=name,
            video=source or "",
            num_frames=len(camera),
            height=height,
            width=width,
            fps=camera.fps,
            intrinsics_per_frame=camera.K,
        )
        if not self.requires_video_path:
            record["_frames_rgb"] = video
        return record, camera

    def _validate_camera(self, camera, inputs):
        """Require a supported camera timeline and fixed image geometry."""
        if not isinstance(camera, Camera):
            raise TypeError("camera must be Camera")
        if not np.isfinite(camera.K).all() or not np.isfinite(camera.fps):
            raise ValueError("Camera intrinsics K and fps must be finite")
        if self.mode == "stream" and (len(camera) != 1 or inputs.get("video") is None):
            raise ValueError(
                "stream requires one RGB image or one video frame and one camera row"
            )
        if not np.all(camera.canvas == camera.canvas[0]):
            raise ValueError("Native ports require a fixed canvas")

    def _prepare_video(self, inputs, camera):
        """Resolve local RGB frames and retain native video paths when required."""
        source = inputs.get("video_path")
        video = inputs.get("video")
        if isinstance(video, (str, Path)):
            source = video
            video = None
        if source is not None:
            source = str(Path(source).resolve())
        if self.requires_video_path and source is None:
            raise ValueError("This native adapter requires the original video path")
        # A path input leaves decoding inside native temporal ports; array input
        # is kept in memory for frame-based ports and never lossy-reencoded.
        if video is None and not self.requires_video_path:
            if source is None:
                raise ValueError("Provide RGB video or video_path")
            video = load_video(source)
        if video is not None:
            video = np.asarray(video)
            if (
                video.dtype != np.uint8
                or video.ndim != 4
                or video.shape[-1] != 3
                or len(video) != len(camera)
            ):
                raise ValueError("Expected aligned uint8 RGB [T,H,W,3]")
            if not np.all(camera.canvas == [video.shape[2], video.shape[1]]):
                raise ValueError("Camera/RGB canvas mismatch")
        return source, video


def manifest_records(path):
    """Read a prepared split manifest and index its records by clip, or (None, {}) without one."""
    path = Path(path)
    if not path.is_file():
        return None, {}
    manifest = read_json(path)
    return manifest, {row["clip_id"]: row for row in manifest.get("records", [])}


def manifest_file(records, name, kind):
    """Return a clip's file of one kind from manifest records (right camera entry); falsy -> caller's default."""
    value = records.get(name, {}).get(kind)
    return value["right"] if isinstance(value, dict) and value else value


class BenchmarkDataset:
    """Local clip inputs resolved through the published split manifest."""

    def _path(self, name, kind):
        value = manifest_file(self.records, name, kind)
        if value:
            return self.root / value
        if kind == "camera":
            return self.root / "cameras_json" / (name + ".json")
        if kind == "mano":
            return self.root / "mano" / (name + ".npz")
        for folder, suffix in (
            ("mp4/rgb_480p", ".mp4"),
            ("mp4", ".mp4"),
            ("png", ".png"),
        ):
            path = self.root / folder / (name + suffix)
            if path.is_file():
                return path
        raise FileNotFoundError(f"No RGB file for {name} in {self.root}")

    def __init__(self, root="datasets/h2o", split="test"):
        self.root = Path(root).expanduser().resolve()
        manifest = self.root / "manifests/split.json"
        data, self.records = manifest_records(manifest)
        if data is None:
            raise FileNotFoundError(manifest)
        self.names = tuple(data[split])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        camera_path = self._path(name, "camera")
        camera = Camera.load(camera_path)
        return dict(
            name=name,
            camera=camera,
            camera_path=str(camera_path),
            video_path=str(self._path(name, "video")),
            mano_path=str(self._path(name, "mano")),
        )
