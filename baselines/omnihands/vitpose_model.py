"""Native ViTPose hand-keypoint inference for the owning baseline."""

import os
import torch
from baselines.omnihands.third_party.ViTPose.mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)

os.environ["PYOPENGL_PLATFORM"] = "egl"


class ViTPoseModel:
    def __init__(self, config, checkpoint, device):
        self.model = init_pose_model(config, checkpoint, device=torch.device(device))

    def predict_pose(self, image, det_results):
        image = image[:, :, ::-1]
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(
            self.model,
            image,
            person_results=person_results,
            bbox_thr=0.5,
            format="xyxy",
        )
        return out
