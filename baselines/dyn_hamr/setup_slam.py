"""Install the pinned upstream DROID frontend in the Dyn-HaMR environment."""

from pathlib import Path
from importlib import invalidate_caches
import subprocess
import sys


def main(assets=None):
    source = Path(__file__).resolve().parent
    droid = source / "third_party/DROID-SLAM"
    revision = "385bffda51190fad66e135ff337ff95fd9c049dc"
    if not droid.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/ZhengdiYu/DROID-SLAM.git", str(droid)],
            check=True,
        )
    subprocess.run(["git", "-C", str(droid), "checkout", revision], check=True)
    subprocess.run(
        ["git", "-C", str(droid), "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    from baselines.dyn_hamr.droid_build import prepare_namespace, prepare_setuptools, prepare_torch_dispatch

    prepare_torch_dispatch(droid)
    prepare_setuptools(droid)
    prepare_namespace(droid)
    try:
        import torch  # noqa: F401
        from baselines.dyn_hamr import droid_backends, lietorch  # noqa: F401
    except (ImportError, OSError):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(droid)],
            check=True,
        )
        invalidate_caches()
        from baselines.dyn_hamr import droid_backends, lietorch  # noqa: F401
    from utils.download_data import checkpoint_dir
    from utils.io import resource_path

    resource_path(assets or checkpoint_dir("dyn_hamr"), "dyn_hamr.droid")


if __name__ == "__main__":
    main()
