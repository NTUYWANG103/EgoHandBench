"""Run explicit InfiniHand refinement/runtime options on an EgoHandBench dataset."""
import argparse
import json
from pathlib import Path

from baselines.hand_detector import HandDetector
from utils.io import BenchmarkDataset, save_npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path, help='Prepared benchmark dataset directory')
    p.add_argument('--output', required=True, type=Path, help='Fresh output directory')
    p.add_argument('--checkpoints', type=Path)
    p.add_argument('--version', choices=['official', 'egohandbench'], default='egohandbench')
    p.add_argument('--branch-precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--no-postprocess', action='store_true')
    p.add_argument('--postprocess-lambda', type=float, default=0.5)
    p.add_argument('--postprocess-storage', choices=['cpu', 'cpu_async', 'cuda'], default='cpu')
    p.add_argument('--cuda-graphs', action='store_true')
    p.add_argument('--fused-rope', action='store_true')
    p.add_argument('--compact-kv-cache', action='store_true')
    p.add_argument('--mask-box-backend', choices=['torch', 'triton'], default='torch')
    p.add_argument('--export-device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--joint-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        raise ValueError('Use a fresh output directory to avoid mixing protocols')
    runtime = dict(postprocess=not a.no_postprocess, postprocess_lambda=a.postprocess_lambda,
                   branch_precision=a.branch_precision, postprocess_storage=a.postprocess_storage,
                   cuda_graphs=a.cuda_graphs, fused_rope=a.fused_rope,
                   compact_kv_cache=a.compact_kv_cache, mask_box_backend=a.mask_box_backend,
                   export_device=a.export_device)
    (a.output/'CONFIG.json').write_text(json.dumps(dict(version=a.version, **runtime), indent=2))
    model = HandDetector('infinihand', version=a.version, checkpoints=a.checkpoints)
    count = 0
    for sample in BenchmarkDataset(a.dataset):
        prediction = model.inference_offline(sample, joint_only=a.joint_only, **runtime)
        for name, arrays in prediction.items():
            if arrays is not None:
                save_npz(a.output/name/(sample['name']+'.npz'), arrays)
        report = a.output/'runtime'/(sample['name']+'.json')
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(model.model.last_report, indent=2))
        count += 1
    (a.output/'SUCCESS.json').write_text(json.dumps(dict(status='complete', clips=count), indent=2))


if __name__ == '__main__':
    main()
