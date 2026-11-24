"""Hugging Face downloads: prepared datasets, baseline checkpoint groups and canonical MANO."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import tempfile

from utils.io import REPO_ROOT, read_json


HUB = "https://huggingface.co"
REPOSITORY = "WANG-XIXI/EgoHandBench"
DATASETS = (
    "arctic",
    "egodemo_egopro_standard",
    "egodemo_egostandard",
    "gen_humanego_1h",
    "h2o",
    "hot3d",
    "oakink2",
)


def copy_file(source, destination):
    """Copy through a per-writer temporary name so an interrupted or concurrent copy never looks complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".partial", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        shutil.copyfile(source, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_file(repository, name, destination=None):
    """Fetch a file from the Hugging Face Hub and optionally copy it to a local path."""
    if destination is not None:
        destination = Path(destination)
        if destination.is_file():
            return destination
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            repository, name, repo_type="dataset", endpoint=HUB
        )
    )
    if destination is None:
        return source
    copy_file(source, destination)
    return destination


def download_folder(repository, prefix, destination, last=None):
    """Download a folder from the Hugging Face Hub, retaining its relative file layout.

    ``last`` (a file name inside the folder) is fetched after every other file, so it can mark the folder complete.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    names = [
        entry.path
        for entry in HfApi(endpoint=HUB).list_repo_tree(
            repository, repo_type="dataset", path_in_repo=prefix, recursive=True
        )
        if isinstance(entry, RepoFile)
    ]
    prefix += "/"
    destination = Path(destination).expanduser().resolve()

    def download(name):
        download_file(repository, name, destination / name[len(prefix) :])

    final = prefix + last if last is not None else None
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(download, [name for name in names if name != final]))
    if final in names:
        download(final)
    return destination


# Files downloaded from their official release instead of this repository.
EXTERNAL_FILES = {
    "detector.pt": dict(
        repository="rolpotamias/WiLoR",
        repo_type="space",
        name="pretrained_models/detector.pt",
        sha256="5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
    ),
}


def download_external(destination, source):
    """Fetch an official release file and verify its SHA-256 before use."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            source["repository"], source["name"],
            repo_type=source["repo_type"], endpoint=HUB,
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != source["sha256"]:
        raise ValueError(f"Checksum mismatch for {source['repository']}/{source['name']}")
    copy_file(path, destination)


def checkpoint_dir(method, checkpoints=None, *, version=None):
    """Resolve or download one complete method/version checkpoint group."""
    configuration = read_json(REPO_ROOT / "baselines" / method / "config.json")
    version = configuration["default"] if version is None else version
    root = (
        Path(checkpoints if checkpoints is not None else "checkpoints")
        .expanduser()
        .resolve()
    )
    folder = root if (root / "layout.json").is_file() else root / method / version
    if checkpoints is not None or (folder / "layout.json").is_file():
        return folder
    # layout.json marks a complete group, so it is written only after every other file.
    entries = configuration["versions"][version]
    for relative in [*entries.get("weights", {}).values(), *entries.get("resources", {}).values()]:
        source = EXTERNAL_FILES.get(Path(relative).name)
        if source is not None and not (folder / relative).is_file():
            download_external(folder / relative, source)
    return download_folder(REPOSITORY, f"baselines/{method}/{version}", folder, last="layout.json")


def canonical_mano_dir():
    """Resolve canonical RIGHT MANO from WiLoR's self-contained asset group."""
    name = "checkpoints/mano/MANO_RIGHT.pkl"
    return download_file(
        REPOSITORY,
        f"baselines/wilor/official/{name}",
        REPO_ROOT / "checkpoints/wilor/official" / name,
    ).parent


def download_dataset(dataset=None):
    """Download all seven datasets, or one dataset, from the Hugging Face Hub."""
    target = Path("datasets").resolve()
    prefix = "datasets"
    if dataset is not None:
        prefix += "/" + dataset
        target /= dataset
    return download_folder(REPOSITORY, prefix, target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download every prepared dataset unless one is selected."
    )
    parser.add_argument("--dataset", choices=DATASETS)
    print(download_dataset(parser.parse_args().dataset))
