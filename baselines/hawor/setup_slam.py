"""Install the pinned upstream HaWoR SLAM (masked DROID-SLAM + Metric3D) and weights."""

from pathlib import Path
from importlib import invalidate_caches
import subprocess
import sys


def main(assets=None):
    from utils.download_data import checkpoint_dir
    from utils.io import resource_path

    source = Path(__file__).resolve().parent / "third_party/HaWoR"
    revision = "66c7d4108d58a716deccd192cb7645170cdc7bd7"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/ThunderVVV/HaWoR.git", str(source)],
            check=True,
        )
    subprocess.run(["git", "-C", str(source), "checkout", revision], check=True)
    subprocess.run(
        ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    from baselines.hawor.droid_build import prepare_namespace, prepare_setuptools, prepare_torch_dispatch

    prepare_torch_dispatch(source / "thirdparty/DROID-SLAM")
    prepare_setuptools(source / "thirdparty/DROID-SLAM")
    prepare_namespace(source / "thirdparty/DROID-SLAM")
    try:
        import torch  # noqa: F401
        from baselines.hawor import droid_backends, lietorch  # noqa: F401
    except (ImportError, OSError):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
             str(source / "thirdparty/DROID-SLAM")],
            check=True,
        )
        invalidate_caches()
        from baselines.hawor import droid_backends, lietorch  # noqa: F401
    assets = assets or checkpoint_dir("hawor")
    for key in ("hawor.droid", "hawor.metric3d", "hawor.infiller"):
        resource_path(assets, key)


if __name__ == "__main__":
    main()
