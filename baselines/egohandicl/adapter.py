"""EgoHandICL: WiLoR coarse MANO refined by in-context templates, VLM cues and Qwen-7B text."""

import random

import cv2
import numpy as np
import torch
from PIL import Image

from utils.download_data import checkpoint_dir
from utils.io import InferenceInput, resource_path, VideoReader, weight_path
from utils.mano import enable_mano_pickle_compatibility, ManoModel, refit_prediction
from .model import HAND_DIM, HandICL
from .vlm import RetrievalCues, TextEncoder

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_image(rgb):
    """Native transform: whole RGB frame -> PIL bilinear Resize((224, 224)), uint8."""
    return np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR))


def normalize_images(images, device):
    """ToTensor + ImageNet Normalize on uint8 [B,224,224,3]."""
    x = torch.as_tensor(np.asarray(images), device=device).float().div_(255)
    mean = torch.as_tensor(IMAGE_MEAN, device=device)
    std = torch.as_tensor(IMAGE_STD, device=device)
    return ((x - mean) / std).permute(0, 3, 1, 2).contiguous()


def hand_vector(global_orient, hand_pose, betas, cam):
    """One hand in the native 61-value layout via cv2.Rodrigues (process_mano)."""
    rotations = np.concatenate(
        [np.asarray(global_orient).reshape(1, 3, 3), np.asarray(hand_pose).reshape(15, 3, 3)]
    )
    axis_angle = [cv2.Rodrigues(np.asarray(r, np.float64))[0].reshape(3) for r in rotations]
    return np.concatenate(
        [np.concatenate(axis_angle), np.asarray(betas).reshape(10), np.asarray(cam).reshape(3)]
    ).astype(np.float32)


def decode_hand(vector):
    """Inverse of hand_vector (aa_to_R_list in released inference)."""
    vector = np.asarray(vector, np.float64)
    rotations = np.stack(
        [cv2.Rodrigues(vector[3 * i:3 * i + 3])[0] for i in range(16)]
    ).astype(np.float32)
    return rotations[:1], rotations[1:], vector[48:58].astype(np.float32), vector[58:61].astype(np.float32)


class CoarseWiLoR:
    """WiLoR detector and regressor that provide EgoHandICL's coarse MANO inputs."""

    def __init__(self, assets, device):
        from ultralytics import YOLO
        from .wilor.datasets.vitdet_dataset import ViTDetDataset
        from .wilor.models import load_wilor
        from .wilor.utils import recursive_to

        self.device = device
        self.dataset_class = ViTDetDataset
        self.recursive_to = recursive_to
        self.model, self.cfg = load_wilor(
            str(weight_path(assets, "wilor")),
            str(resource_path(assets, "wilor.config")),
            mano_dir=str(resource_path(assets, "wilor.mano")),
        )
        self.model = self.model.float().to(device).eval()
        self.detector = YOLO(str(resource_path(assets, "wilor.detector"))).to(device)

    @torch.inference_mode()
    def __call__(self, frames_bgr, batch_size=16):
        """Per frame and side [left, right]: native WiLoR outputs or None."""
        samples, locations = [], []
        for fi, image in enumerate(frames_bgr):
            boxes = self.detector(image, conf=0.3, verbose=False)[0].boxes.data.cpu().numpy()
            chosen = []
            for side in (0, 1):
                candidates = boxes[boxes[:, 5].astype(int) == side]
                if len(candidates):
                    chosen.append(candidates[np.argmax(candidates[:, 4])])
            if not chosen:
                continue
            chosen = np.stack(chosen)
            dataset = self.dataset_class(
                self.cfg, image, chosen[:, :4], chosen[:, 5], rescale_factor=2.0, fp16=False
            )
            for i in range(len(dataset)):
                samples.append(dataset[i])
                locations.append((fi, int(chosen[i, 5])))
        outputs = [[None, None] for _ in frames_bgr]
        for start in range(0, len(samples), batch_size):
            batch = torch.utils.data.default_collate(samples[start:start + batch_size])
            batch = self.recursive_to(batch, self.device)
            pred = self.model(batch)
            params = {k: v.float().cpu().numpy() for k, v in pred["pred_mano_params"].items()}
            cam = pred["pred_cam"].float().cpu().numpy()
            for i, (fi, side) in enumerate(locations[start:start + batch_size]):
                hand = dict(
                    global_orient=params["global_orient"][i].reshape(1, 3, 3),
                    hand_pose=params["hand_pose"][i].reshape(15, 3, 3),
                    betas=params["betas"][i].reshape(10),
                    cam=cam[i],
                    box_center=batch["box_center"][i].float().cpu().numpy(),
                    box_size=float(batch["box_size"][i]),
                    img_size=batch["img_size"][i].float().cpu().numpy(),
                )
                if not all(np.isfinite(v).all() for v in hand.values()):
                    raise ValueError("Non-finite model prediction")
                outputs[fi][side] = hand
        return outputs


def coarse_vector(hands):
    """Both hands' coarse 122-vector; an undetected hand is all zeros."""
    return np.concatenate(
        [
            hand_vector(h["global_orient"], h["hand_pose"], h["betas"], h["cam"])
            if h is not None else np.zeros(HAND_DIM, np.float32)
            for h in hands
        ]
    )


def pad_states(states, device):
    """Stack variable-length text states; True marks padding (ignored keys)."""
    length = max(len(s) for s in states)
    padded = torch.zeros(len(states), length, states[0].shape[-1], device=device, dtype=states[0].dtype)
    mask = torch.ones(len(states), length, dtype=torch.bool, device=device)
    for i, s in enumerate(states):
        padded[i, : len(s)] = s
        mask[i, : len(s)] = False
    return padded, mask


class EgoHandICLAdapter:

    def __init__(self, checkpoints=None, device=None, seed=179, batch_size=None, *, version="egohandbench"):
        self.method = "egohandicl"
        selected = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if selected.type == "cuda" and selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        if selected.type != "cuda":
            raise ValueError("egohandicl requires CUDA")
        self.device = str(selected)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        self.seed = seed
        self.batch_size = 16 if batch_size is None else batch_size
        self.assets = checkpoint_dir(self.method, checkpoints, version=version)
        self._mano_model = None
        enable_mano_pickle_compatibility()
        self.reset()
        self.coarse = CoarseWiLoR(self.assets, self.device)
        self.cues = RetrievalCues(str(resource_path(self.assets, "egohandicl.vlm")), self.device)
        self.text = TextEncoder(str(resource_path(self.assets, "egohandicl.text")), self.device)
        state = torch.load(weight_path(self.assets, "egohandicl"), map_location="cpu")
        self.icl = HandICL(**state["config"])
        self.icl.load_state_dict(state["model_state_dict"])
        self.icl = self.icl.to(self.device).eval()
        templates = np.load(resource_path(self.assets, "egohandicl.templates"), allow_pickle=False)
        self.templates = {k: templates[k] for k in templates.files}
        self._text_cache = {}
        self.reset()

    def inference_stream(self, inputs, *, joint_only=False):
        """Predict one calibrated RGB frame; template draws continue the seeded stream until reset."""
        return self._inference(inputs, "stream", joint_only)

    def inference_offline(self, inputs, *, joint_only=False):
        return self._inference(inputs, "offline", joint_only)

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

    def _inference(self, inputs, mode, joint_only):
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        # PyTorch defaults; the released inference sets no TF32 flag (as in the WiLoR adapter)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        with InferenceInput(inputs, mode=mode) as (record, camera):
            raw = self.predict(record)
            if not joint_only and self._mano_model is None:
                self._mano_model = ManoModel(resource_path(self.assets, "mano"))
            return refit_prediction(raw, camera, self._mano_model, joint_only=joint_only)

    def _states(self, text, cache=False):
        """Qwen-7B states of one caption; only the fixed template captions are kept on the GPU."""
        if not cache:
            return self.text(text)
        if text not in self._text_cache:
            self._text_cache[text] = self.text(text)
        return self._text_cache[text]

    def _template_indices(self, types):
        """Random template of the query's hand-involvement type (released random.choice)."""
        pool = self.templates["type"]
        chosen = []
        for t in types:
            candidates = np.flatnonzero(pool == t)
            if not len(candidates):
                candidates = np.arange(len(pool))
            chosen.append(int(candidates[random.randrange(len(candidates))]))
        return np.asarray(chosen)

    @torch.no_grad()
    def refine(self, images224, coarse, types, captions):
        """EgoHandICL forward for query frames; returns predicted m2b 122-vectors."""
        index = self._template_indices(types)
        query = torch.as_tensor(np.asarray(coarse), device=self.device)
        text1, mask1 = pad_states(
            [self._states(str(self.templates["caption"][i]), cache=True) for i in index], self.device
        )
        text2, mask2 = pad_states([self._states(c) for c in captions], self.device)
        batch = dict(
            img1=normalize_images(self.templates["image"][index], self.device),
            img2=normalize_images(images224, self.device),
            text1=text1, text1_mask=mask1, text2=text2, text2_mask=mask2,
            mano_params=dict(
                m1a=torch.as_tensor(self.templates["m1a"][index], device=self.device),
                m1b=torch.as_tensor(self.templates["m1b"][index], device=self.device),
                m2a=query,
                m2b=torch.randn_like(query) * 1e-5,
            ),
        )
        with torch.autocast("cuda", dtype=torch.float16):
            output = self.icl(batch)
        return output[:, -2 * HAND_DIM:].float().cpu().numpy()

    def predict(self, record):
        n, width, height = record["num_frames"], record["width"], record["height"]
        cap = VideoReader(record["_frames_rgb"], color="rgb")
        chunks = []
        for begin in range(0, n, self.batch_size):
            frames = []
            for _ in range(min(self.batch_size, n - begin)):
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (height, width):
                    raise ValueError("Input timeline/size mismatch")
                frames.append(frame)
            chunks.append(self.cue_chunk(frames))
        if cap.read()[0]:
            raise ValueError("Unexpected extra RGB frames")
        return self.export(chunks, n, width, height)

    def cue_chunk(self, frames):
        """Coarse WiLoR hands, VLM cues and 224 images for one chunk of RGB frames."""
        hands = self.coarse([f[..., ::-1].copy() for f in frames])
        rows = [j for j, h in enumerate(hands) if h[0] is not None or h[1] is not None]
        types, captions = (
            self.cues([frames[j] for j in rows], batch_size=self.batch_size) if rows else ([], [])
        )
        return dict(
            hands=hands, rows=rows, types=list(types), captions=list(captions),
            images=[resize_image(frames[j]) for j in rows],
        )

    def export(self, chunks, n, width, height):
        """Refine each chunk in order and place hands with the coarse WiLoR camera."""
        raw = {
            k: np.full((n, 2, *tail), np.nan, np.float32)
            for k, tail in [
                ("vertices_cam", (778, 3)),
                ("joints_cam", (21, 3)),
                ("global_orient", (1, 3, 3)),
                ("hand_pose", (15, 3, 3)),
                ("betas", (10,)),
                ("cam_trans", (3,)),
            ]
        }
        raw["presence"] = np.zeros((n, 2), bool)
        focal = float(self.coarse.cfg.EXTRA.FOCAL_LENGTH) / int(self.coarse.cfg.MODEL.IMAGE_SIZE) * max(width, height)
        begin = 0
        for chunk in chunks:
            hands, rows = chunk["hands"], chunk["rows"]
            if rows:
                refined = self.refine(
                    chunk["images"], [coarse_vector(hands[j]) for j in rows],
                    chunk["types"], chunk["captions"],
                )
                items = [(r, j, side) for r, j in enumerate(rows) for side in (0, 1) if hands[j][side] is not None]
                decoded = [decode_hand(refined[r, side * HAND_DIM:(side + 1) * HAND_DIM]) for r, j, side in items]
                out = self.coarse.model.mano(
                    global_orient=torch.as_tensor(np.stack([d[0] for d in decoded]), device=self.device),
                    hand_pose=torch.as_tensor(np.stack([d[1] for d in decoded]), device=self.device),
                    betas=torch.as_tensor(np.stack([d[2] for d in decoded]), device=self.device),
                    pose2rot=False,
                )
                vertices = out.vertices.float().cpu().numpy()
                joints = out.joints.float().cpu().numpy()
                for (r, j, side), (go, pose, betas, _), verts, joint in zip(items, decoded, vertices, joints):
                    hand = hands[j][side]
                    # Placement keeps the coarse WiLoR crop camera (EgoHandICL's losses
                    # do not supervise its cam entries), converted as in WiLoR.
                    cam = hand["cam"].astype(np.float64).copy()
                    cam[1] *= 2 * side - 1
                    bs = hand["box_size"] * cam[0] + 1e-9
                    trans = np.array([
                        2 * (hand["box_center"][0] - width / 2.0) / bs + cam[1],
                        2 * (hand["box_center"][1] - height / 2.0) / bs + cam[2],
                        2 * focal / bs,
                    ], np.float32)
                    mirror = 2 * side - 1
                    verts[:, 0] *= mirror
                    joint[:, 0] *= mirror
                    t = begin + j
                    raw["global_orient"][t, side] = go
                    raw["hand_pose"][t, side] = pose
                    raw["betas"][t, side] = betas
                    raw["cam_trans"][t, side] = trans
                    raw["vertices_cam"][t, side] = verts + trans
                    raw["joints_cam"][t, side] = joint + trans
                    raw["presence"][t, side] = True
            begin += len(hands)
        if begin != n:
            raise ValueError("Chunk/timeline mismatch")
        for key in ("vertices_cam", "joints_cam", "cam_trans"):
            if not np.isfinite(raw[key][raw["presence"]]).all():
                raise ValueError("Non-finite model prediction")
        raw.update(
            frame_ids=np.arange(n),
            projection_intrinsics=np.array([focal, focal, width / 2, height / 2], np.float64),
        )
        return raw
