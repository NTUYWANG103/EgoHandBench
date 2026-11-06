"""Run a baseline on a prepared local dataset and save its MANO and its joints.

Writes ``<output>/mano/<clip>.npz`` (MANO parameters) and
``<output>/joint/<clip>.npz`` (``joints3d`` [F, 2, 21, 3]: the scored joints,
camera space, MANO16 + five tips); ``<output>`` defaults to
``results/<dataset>/<method>/<version>``.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from baselines.hand_detector import HandDetector
from utils.io import BenchmarkDataset, save_npz


def run(method="wilor", dataset="h2o", output=None, *, version=None, **kwargs):
    model = HandDetector(method, version=version)
    output = Path(output) if output else Path("results") / dataset / method / model.version
    for sample in tqdm(
        BenchmarkDataset(Path("datasets") / dataset),
        desc=f"{dataset}/{method}/{model.version}",
        unit="clip",
    ):
        prediction = model.inference_offline(sample, **kwargs)
        for source, arrays in prediction.items():
            if arrays is not None:
                save_npz(output / source / (sample["name"] + ".npz"), arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="wilor")
    parser.add_argument("--version", help="Model version (defaults to the method's preset)")
    parser.add_argument("--dataset", default="h2o")
    parser.add_argument("--joint-only", action="store_true", help="Save joints only (methods whose joints are decoded from MANO still fit it)")
    parser.add_argument(
        "--output", help="Method results directory (mano/ and joint/ inside)"
    )
    args = parser.parse_args()
    run(
        args.method,
        args.dataset,
        args.output,
        version=args.version,
        joint_only=args.joint_only,
    )
