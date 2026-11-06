"""Compatibility preparation for the pinned DROID/lietorch sources."""

from pathlib import Path
import os
import re


def _replace_source(path, text):
    """Write through a temporary file and rename, so hardlinked copies stay unchanged."""
    temporary = path.with_name(path.name + ".egohandbench-tmp")
    temporary.write_text(text)
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)


def prepare_torch_dispatch(source):
    """Use ScalarType in dispatch macros, as required by PyTorch 2.6.

    Both pinned frontends contain the same four extension sources. Check the
    expected call counts so an upstream change cannot be silently rewritten.
    Repeated setup calls accept the already patched form.
    """
    source = Path(source)
    files = {
        "src/correlation_kernels.cu": 2,
        "src/altcorr_kernel.cu": 1,
        "thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp": 19,
        "thirdparty/lietorch/lietorch/src/lietorch_gpu.cu": 19,
    }
    pattern = re.compile(r"\b(volume|fmap1|a|X)\.(type|scalar_type)\(\)")
    prepared = []
    for relative, expected in files.items():
        path = source / relative
        original = path.read_text()
        matches = list(pattern.finditer(original))
        if len(matches) != expected:
            raise RuntimeError(f"Unexpected pinned DROID dispatch source: {path}")
        updated = pattern.sub(lambda m: f"{m[1]}.scalar_type()", original)
        prepared.append((path, original, updated))
    for path, original, updated in prepared:
        if updated != original:
            _replace_source(path, updated)


def prepare_setuptools(source):
    """Build both extensions and lietorch in this method's own namespace."""
    path = Path(source) / "setup.py"
    original = path.read_text()
    marker = "# EgoHandBench: package both DROID extensions in one wheel.\n"
    updated = original
    if not original.startswith(marker):
        if len(re.findall(r"(?m)^setup\(", original)) != 2:
            raise RuntimeError(f"Unexpected pinned DROID setup script: {path}")
        names = iter(("droid_config", "lietorch_config"))
        updated = re.sub(r"(?m)^setup\(", lambda _: next(names) + " = dict(", original)
        updated += ("\nlietorch_config['ext_modules'] = "
                    "droid_config['ext_modules'] + lietorch_config['ext_modules']\n"
                    "setup(**lietorch_config)\n")
        updated = marker + updated
    # Build and package destinations of the extension wheels.
    namespace = "baselines.dyn_hamr"
    replacements = {
        "name='droid_backends'": "name='egohandbench_dyn_hamr_droid'",
        "CUDAExtension('droid_backends'": f"CUDAExtension('{namespace}.droid_backends'",
        "name='lietorch'": "name='egohandbench_dyn_hamr_slam'",
        "packages=['lietorch']": f"packages=['{namespace}.lietorch']",
        "package_dir={'': 'thirdparty/lietorch'}":
            f"package_dir={{'{namespace}.lietorch': 'thirdparty/lietorch/lietorch'}}",
        "CUDAExtension('lietorch_backends'": f"CUDAExtension('{namespace}.lietorch_backends'",
    }
    for before, after in replacements.items():
        if updated.count(before) == 1:
            updated = updated.replace(before, after)
        elif updated.count(after) != 1:
            raise RuntimeError(f"Unexpected pinned DROID package declaration: {path}")
    if updated != original:
        _replace_source(path, updated)


def prepare_namespace(source):
    """Persist qualified imports after a fresh clone, without runtime aliases."""
    namespace = "baselines.dyn_hamr"
    source = Path(source)
    paths = [*source.glob("*.py"),
             *(source / "droid_slam").rglob("*.py"),
             *(source / "thirdparty/lietorch/lietorch").glob("*.py")]
    for path in paths:
        original = path.read_text()
        updated = re.sub(
            r"(?m)^([ \t]*)import (droid_backends|lietorch_backends|lietorch)[ \t]*$",
            lambda match: f"{match[1]}from {namespace} import {match[2]}", original,
        )
        updated = re.sub(
            r"(?m)^([ \t]*)from lietorch(?= import|\.)",
            lambda match: f"{match[1]}from {namespace}.lietorch", updated,
        )
        if updated != original:
            _replace_source(path, updated)
