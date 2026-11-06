#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
import pickle
import re
import sys
import time
from typing import Dict, List

import numpy as np
import torch



_MT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_train
_VENDOR = os.path.join(_MT, "_vendor")

from baselines.mint.model_train.core.registry import DATASETS
from baselines.mint.model_train.data.base_dataset import BaseClipDataset
from baselines.mint.model_train.data.transforms import preprocess_frames
from baselines.mint.model_train.losses.normalization import resolved_camera_translation_scales
from baselines.mint.model_train._vendor.lingbot_map.utils.rotation import mat_to_quat, quat_to_mat


def _col_to_np(table, name) -> np.ndarray:
    """Internal helper."""
    return np.array(table.column(name).to_pylist())


def _rebase_pose_enc_to_first(pe: torch.Tensor) -> torch.Tensor:
    T, quat, fov = pe[:, :3], pe[:, 3:7], pe[:, 7:]
    R = quat_to_mat(quat)                                   # [S,3,3]
    R0t = R[0].transpose(-1, -2)
    Rn = R @ R0t
    Tn = T - torch.einsum("sij,j->si", Rn, T[0])
    qn = mat_to_quat(Rn)                                    # [S,4] xyzw
    return torch.cat([Tn, qn, fov], dim=-1)



_MANO_COLS = ["transl_cam", "orient6d", "pose6d", "betas"]
_INDEX_CACHE_VERSION = 2


def _episode_local_start(seg: dict, file_rows: int) -> int:
    """Resolve an episode's local row range from authoritative episode metadata."""
    local0 = int(seg["dataset_from"] - seg["data_file_from"])
    local1 = local0 + int(seg["length"])
    if local0 < 0 or local1 > file_rows:
        raise RuntimeError(
            f"[train]  {seg['episode']}."
            f"file={seg['data_parquet']} local=[{local0},{local1}) rows={file_rows};"
            "[train]"
        )
    return local0


@DATASETS.register("lerobot_v3")
class LeRobotV3Dataset(BaseClipDataset):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.root = cfg["root"]
        info = json.load(open(os.path.join(self.root, "meta", "info.json")))
        self.fps = info.get("fps", 30.0)

        ego = info["features"]["observation.images.ego"]["shape"]  # [H, W, 3]
        self.orig_hw = (int(ego[0]), int(ego[1]))
        self.video_key = cfg.get("video_key", "observation.images.ego")
        self.max_segments = cfg.get("max_segments")
        self.stride = int(cfg.get("clip_stride", 1))
        self.require_mano_gt = bool(cfg.get("require_mano_gt", True))
        self.require_kpt21_gt = bool(
            cfg.get("require_kpt21_gt", cfg.get("require_kpt21", False))
        )
        self.camera_trans_scales = resolved_camera_translation_scales(cfg)
        self._cache = {}
        self.segments = self._build_index()
        if not self.segments:
            raise RuntimeError(self._empty_index_hint())


        self.clips = []
        for si, seg in enumerate(self.segments):
            last = seg["length"] - self.clip_len
            for off in range(0, last + 1, self.stride):
                self.clips.append((si, off))

    def _empty_index_hint(self) -> str:
        """Internal helper."""
        vdir = os.path.join(self.root, "videos", self.video_key)
        vids = glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True)[:20]
        dangling = [p for p in vids if os.path.islink(p) and not os.path.exists(p)]
        base = f"[train]  {self.root}; {self.clip_len}."
        if vids and len(dangling) == len(vids):
            tgt = os.path.realpath(vids[0])
            return (f"[train]  {base}."
                    f"[train]  {tgt}."
                    f"[train]")
        return base





    def _index_cache_path(self, mfiles) -> str:
        vks = re.sub(r"[^0-9A-Za-z]+", "_", self.video_key)
        return os.path.join(self.root, "meta", f".clip_index.cl{self.clip_len}.{vks}.pkl")

    @staticmethod
    def _meta_sig(mfiles) -> list:

        return [len(mfiles), max(os.path.getmtime(f) for f in mfiles)]

    def _build_index(self) -> List[dict]:
        mfiles = sorted(glob.glob(
            os.path.join(self.root, "meta", "episodes", "**", "*.parquet"), recursive=True))
        if not mfiles:
            raise RuntimeError(f"[train]  {self.root}.")

        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        is_rank0 = rank == 0
        cache = self._index_cache_path(mfiles)
        sig = self._meta_sig(mfiles)

        def _load_cache():
            try:
                with open(cache, "rb") as f:
                    obj = pickle.load(f)
            except Exception:
                return None


            if (obj.get("version") == _INDEX_CACHE_VERSION
                    and obj.get("sig") == sig and obj.get("max_segments") == self.max_segments
                    and obj.get("pathmode") == "rel"):
                return obj["segments"]
            return None


        segs = _load_cache()
        if segs is not None:
            if is_rank0:
                print(f"[train]  {len(segs)}; {cache}.")
            return segs


        if world > 1 and not is_rank0:
            for _ in range(1800):
                time.sleep(1.0)
                segs = _load_cache()
                if segs is not None:
                    return segs



        segs = self._scan_meta(mfiles, show_progress=is_rank0)



        if is_rank0 and segs:
            try:
                tmp = f"{cache}.tmp.{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump({"version": _INDEX_CACHE_VERSION,
                                 "sig": sig, "max_segments": self.max_segments,
                                 "pathmode": "rel", "segments": segs},
                                f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache)
                print(f"[train]  {len(segs)}; {cache}.")
            except Exception as e:
                print(f"[train]  {e}.")
        return segs

    def _scan_meta(self, mfiles, show_progress: bool) -> List[dict]:
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            from tqdm import tqdm
        except Exception:
            def tqdm(x, **k):
                return x

        vk = self.video_key
        need = ["episode_index", "length",
                "data/chunk_index", "data/file_index",
                f"videos/{vk}/chunk_index", f"videos/{vk}/file_index",
                f"videos/{vk}/from_timestamp", "dataset_from_index"]

        def _read_one(mf):
            t = pq.read_table(mf, columns=need)
            return {c: t.column(c).to_numpy(zero_copy_only=False) for c in need}


        results = [None] * len(mfiles)
        workers = min(32, max(4, len(mfiles)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_read_one, mf): i for i, mf in enumerate(mfiles)}
            for fut in tqdm(as_completed(futs), total=len(mfiles),
                            disable=not show_progress, desc="[train]",
                            unit="file"):
                results[futs[fut]] = fut.result()

        arr = {c: np.concatenate([r[c] for r in results]) for c in need}




        all_dci = arr["data/chunk_index"].astype(np.int64)
        all_dfi = arr["data/file_index"].astype(np.int64)
        all_dsfrom = arr["dataset_from_index"].astype(np.int64)
        file_key_scale = int(all_dfi.max()) + 1
        all_file_keys = all_dci * file_key_scale + all_dfi
        file_from = np.full(int(all_file_keys.max()) + 1, np.iinfo(np.int64).max,
                            dtype=np.int64)
        np.minimum.at(file_from, all_file_keys, all_dsfrom)


        length = arr["length"].astype(np.int64)
        keep = np.nonzero(length >= self.clip_len)[0]
        dci = all_dci[keep].astype(int)
        dfi = all_dfi[keep].astype(int)
        vci = arr[f"videos/{vk}/chunk_index"][keep].astype(int)
        vfi = arr[f"videos/{vk}/file_index"][keep].astype(int)
        from_ts = arr[f"videos/{vk}/from_timestamp"][keep].astype(float)
        dsfrom = all_dsfrom[keep]
        dfile_from = file_from[all_file_keys[keep]]
        epi = arr["episode_index"][keep].astype(np.int64)
        lens = length[keep]


        def _dpath(ci, fi):
            return os.path.join("data", f"chunk-{ci:03d}", f"file-{fi:03d}.parquet")

        def _vpath(ci, fi):
            return os.path.join("videos", vk, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")


        exist_d = {k for k in {(int(a), int(b)) for a, b in zip(dci, dfi)}
                   if os.path.exists(os.path.join(self.root, _dpath(*k)))}
        exist_v = {k for k in {(int(a), int(b)) for a, b in zip(vci, vfi)}
                   if os.path.exists(os.path.join(self.root, _vpath(*k)))}

        segs = []
        for j in range(len(keep)):
            dk, vkk = (int(dci[j]), int(dfi[j])), (int(vci[j]), int(vfi[j]))
            if dk not in exist_d or vkk not in exist_v:
                continue
            segs.append({
                "data_parquet": _dpath(*dk),
                "video": _vpath(*vkk),
                "length": int(lens[j]),
                "video_start": int(round(float(from_ts[j]) * self.fps)),
                "dataset_from": int(dsfrom[j]),
                "data_file_from": int(dfile_from[j]),
                "episode": int(epi[j]),
            })
            if self.max_segments and len(segs) >= self.max_segments:
                break
        return segs

    def _file_cols(self, parquet_path: str) -> Dict[str, np.ndarray]:

        if parquet_path not in self._cache:
            import pyarrow.parquet as pq
            self._cache = {}
            abs_path = os.path.join(self.root, parquet_path)
            base_cols = ["state_mask", "hand_kept", "cam_trans", "cam_quat", "cam_fov"]
            mano_cols = [f"{s}_mano_{c}" for s in ("left", "right") for c in _MANO_COLS]
            kpt21_cols = ["left_kpt21", "right_kpt21"]
            cols = base_cols
            if self.require_mano_gt:
                cols += mano_cols
            if self.require_kpt21_gt:
                cols += kpt21_cols
            names = set(pq.read_schema(abs_path).names)
            missing = [c for c in cols if c not in names]
            if missing:
                raise RuntimeError(
                    f"[train]  {abs_path}; {missing}."
                    f"[train]")
            t = pq.read_table(abs_path, columns=cols)
            self._cache[parquet_path] = {c: _col_to_np(t, c) for c in cols}
        return self._cache[parquet_path]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        from decord import VideoReader

        seg_idx, off = self.clips[idx]
        seg = self.segments[seg_idx]

        cols = self._file_cols(seg["data_parquet"])

        local0 = _episode_local_start(seg, len(cols["state_mask"]))
        rows = list(range(local0 + off, local0 + off + self.clip_len))

        state_mask = torch.from_numpy(cols["state_mask"][local0 + off].astype(bool))   # [2]


        gt_pose_enc = torch.from_numpy(np.concatenate(
            [cols["cam_trans"][rows], cols["cam_quat"][rows], cols["cam_fov"][rows]],
            axis=-1)).float()                                                   # [S,9]


        gt_pose_enc = _rebase_pose_enc_to_first(gt_pose_enc)



        def _hand(side: str) -> np.ndarray:
            return np.concatenate([cols[f"{side}_mano_{c}"][rows] for c in _MANO_COLS], axis=-1)

        hand_kept = torch.from_numpy(cols["hand_kept"][rows].astype(bool))      # [S,2]
        if self.require_mano_gt:
            hand_gt = torch.from_numpy(
                np.concatenate([_hand("left"), _hand("right")], axis=-1)
            ).float()
        else:
            hand_gt = torch.zeros(self.clip_len, 2 * 109, dtype=torch.float32)
        if self.require_kpt21_gt:
            kpt21_gt = torch.from_numpy(
                np.stack([cols["left_kpt21"][rows], cols["right_kpt21"][rows]], axis=1)
            ).float().reshape(self.clip_len, 2, 21, 3)
        else:
            kpt21_gt = torch.zeros(self.clip_len, 2, 21, 3, dtype=torch.float32)
        mano_gt_valid = torch.tensor(self.require_mano_gt)
        kpt21_gt_valid = torch.tensor(self.require_kpt21_gt)
        hand_valid = torch.tensor(bool(hand_kept.any()) and (
            self.require_mano_gt or self.require_kpt21_gt
        ))

        vr = VideoReader(os.path.join(self.root, seg["video"]), num_threads=1)

        nframes = len(vr)
        vframes = [min(seg["video_start"] + off + k, nframes - 1) for k in range(self.clip_len)]
        frames = vr.get_batch(vframes).asnumpy()     # [S, H0, W0, 3] uint8 RGB
        images = preprocess_frames(frames, self.size_hw)  # [S,3,H,W]

        sample = {
            "images": images,
            "gt_pose_enc": gt_pose_enc,
            "state_mask": state_mask,
            "hand_gt": hand_gt,
            "hand_kept": hand_kept,
            "hand_valid": hand_valid,
            "mano_gt_valid": mano_gt_valid,
            "kpt21_gt": kpt21_gt,
            "kpt21_gt_valid": kpt21_gt_valid,
        }
        sample.update(self.camera_trans_scales)
        return sample
