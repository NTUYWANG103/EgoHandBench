"""Public evaluation of hand predictions: canonical MANO or camera-space joints."""

import numpy as np

from .utils import (
    _CameraMano,
    MetricKernel,
    PARAM_KEYS,
    metrics_from_buffers,
    table_row,
)
from utils.mano import ManoModel


class Metrics:
    """Evaluate hand predictions against the canonical MANO GT.

    A prediction is either MANO (``global_orient``, ``hand_pose``, ``betas``,
    ``cam_trans``; right / mirrored-left convention) or joints
    (``joints3d`` [F, 2, 21, 3] camera-space metres, left then right, MANO16 +
    the five fingertips -- the order the GT MANO decodes to). Both
    carry ``valid`` [F, 2] (the hand claims) and ``source_frame_ids``. Every
    column is computed from the joints (decoded for a MANO prediction); CT-p
    (camera translation) exists only for a MANO prediction.

    ``score`` returns one clip's metrics. For a dataset, collect ``samples``
    from each clip and pass them to ``summarize`` to pool the measured hands,
    joints, and temporal windows.
    """

    def __init__(self):
        self.model = _CameraMano(ManoModel())
        self._kernel = MetricKernel(mano_layers=self.model)

    def samples(self, prediction, ground_truth, camera):
        """Validate matching MANO/camera timelines and return metric samples."""
        if len(prediction["valid"]) != len(ground_truth["valid"]) or len(camera) != len(
            ground_truth["valid"]
        ):
            raise ValueError("Prediction, GT and camera timelines differ")
        if not (
            np.array_equal(
                prediction["source_frame_ids"], ground_truth["source_frame_ids"]
            )
            and np.array_equal(
                camera.source_frame_ids, ground_truth["source_frame_ids"]
            )
        ):
            raise ValueError("Source frame IDs differ")
        if not np.all(camera.canvas == camera.canvas[0]):
            raise ValueError("Clip scoring requires a fixed canvas")
        matrix = camera.K[0] if np.all(camera.K == camera.K[0]) else camera.K
        intrinsics = dict(
            fx=matrix[..., 0, 0], fy=matrix[..., 1, 1],
            cx=matrix[..., 0, 2], cy=matrix[..., 1, 2]
        )
        if np.any(matrix[..., 0, 1]) or np.any(matrix[..., 1, 0]):
            intrinsics.update(skew_xy=matrix[..., 0, 1], skew_yx=matrix[..., 1, 0])
        scores = prediction["valid"].astype(np.float32)
        reference_parameters = {key: ground_truth[key] for key in PARAM_KEYS}
        reference_parameters["valid"] = ground_truth["valid"]
        if all(key in prediction for key in PARAM_KEYS):
            predicted_parameters = {key: prediction[key] for key in PARAM_KEYS}
            predicted_parameters["vis_hand"] = scores
            return self._kernel.evaluate_mano_clip(
                predicted_parameters, reference_parameters, intrinsics, camera.canvas[0]
            )
        if "joints3d" not in prediction:
            raise ValueError("A prediction holds MANO parameters or joints3d")
        joints = np.asarray(prediction["joints3d"])
        reference_joints = self.model.mano2joint(reference_parameters)
        return self._kernel.evaluate_joint_clip(
            joints, scores, reference_joints, intrinsics, camera.canvas[0]
        )[0]

    def score(self, prediction, ground_truth, camera):
        """Return the metric table for one clip."""
        return self.summarize([self.samples(prediction, ground_truth, camera)])

    @staticmethod
    def summarize(buffers):
        """Pool clip samples and calculate the metric table."""
        if not buffers:
            raise ValueError("No metric samples")
        merged = {
            key: np.concatenate([buffer[key] for buffer in buffers])
            for key in buffers[0]
        }
        summary = metrics_from_buffers(merged)
        table = table_row(summary)
        detection = summary["hand_detection"]["total"]
        table.update(
            Recall=detection["recall"],
            Precision=detection["precision"],
            tp=detection["tp"],
            **{
                "CT-p": summary["cam_trans_error_penalty_m"],   # MANO predictions only (NaN -> None for joints)
                "CT-p-mm": summary["cam_trans_error_penalty_m"] * 1000,
                "Wrist3D-p": summary["wrist_pos_error_penalty_mm"],
                "counts_by_side": merged["det_counts"].sum(0).tolist(),
            },
        )
        return {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in table.items()
        }


def main():
    """Score saved local predictions without loading an inference model."""
    import argparse
    from pathlib import Path
    from tqdm import tqdm
    from utils.io import Camera, load_npz, manifest_file, manifest_records, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="GT MANO directory")
    parser.add_argument(
        "--pred", type=Path, required=True,
        help="Prediction NPZ directory (MANO parameters or joints)",
    )
    parser.add_argument(
        "--camera",
        type=Path,
        help="Camera JSON directory; defaults to GT's sibling cameras_json/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Metrics JSON file; defaults to metrics_<prediction folder>.json beside it",
    )
    args = parser.parse_args()
    manifest, records = manifest_records(args.gt.parent / "manifests/split.json")
    evaluator = Metrics()
    cameras = args.camera or args.gt.parent / "cameras_json"
    samples = []
    paths = (
        [args.pred / (name + ".npz") for name in manifest["test"]]
        if manifest is not None
        else sorted(args.pred.glob("*.npz"))
    )
    for path in tqdm(paths, desc="Evaluating", unit="clip"):
        names = {"mano": path.name, "camera": path.stem + ".json"}
        for kind in names:
            value = manifest_file(records, path.stem, kind)
            if value:
                names[kind] = Path(value).name
        samples.append(
            evaluator.samples(
                load_npz(path),
                load_npz(args.gt / names["mano"]),
                Camera.load(cameras / names["camera"]),
            )
        )
    table = evaluator.summarize(samples)
    output = args.output or args.pred.parent / f"metrics_{args.pred.name}.json"
    write_json(output, table)
    print(output)
    print(table)


if __name__ == "__main__":
    main()
