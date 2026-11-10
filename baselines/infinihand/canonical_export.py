"""Optional accelerated official EgoHandBench export (keeps fitting objective).

GPU fitting is numerically close, not bitwise identical to CPU LBFGS. Use the
CPU default when reproducing the published evaluator exactly.
"""
from pathlib import Path
import torch
from utils.mano import ManoModel, mano_joint_prediction
from baselines.infinihand.mano import native_prediction


def export_canonical(raw, camera, mano_dir, *, device='cpu'):
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        model = ManoModel(Path(mano_dir), device=device)
        parameters = native_prediction(raw, camera, model)
        return {'mano': parameters, 'joint': mano_joint_prediction(parameters, model)}
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
