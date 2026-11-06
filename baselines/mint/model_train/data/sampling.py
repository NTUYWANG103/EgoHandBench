#!/usr/bin/env python3
"""Deterministic helpers for retaining a configured fraction of each dataset."""
import hashlib
import math


def validate_sample_fraction(value, *, context: str = "sample_fraction") -> float:
    """Return a finite fraction in [0, 1] with a useful config error."""
    if isinstance(value, bool):
        raise TypeError(f"{context} must be a number in [0, 1], not bool")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must be a number in [0, 1]") from error
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{context} must be a finite number in [0, 1], got {value!r}")
    return fraction


def retained_sample_count(total: int, fraction: float) -> int:
    """Round to the nearest sample while preserving non-empty positive fractions."""
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    fraction = validate_sample_fraction(fraction)
    if total == 0 or fraction == 0.0:
        return 0
    if fraction == 1.0:
        return total
    return min(total, max(1, int(math.floor(total * fraction + 0.5))))


def dataset_sample_seed(base_seed: int, identity: str) -> int:
    """Derive independent, stable NumPy seeds for different dataset roots."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError(f"sample_seed must be an integer, got {base_seed!r}")
    encoded = f"{base_seed}\0{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little", signed=False)


def deterministic_subset_indices(total: int, keep: int, seed: int) -> list[int]:
    """Select sorted indices without replacement so every DDP rank sees one subset."""
    import numpy as np

    total, keep = int(total), int(keep)
    if not 0 <= keep <= total:
        raise ValueError(f"keep must be in [0, {total}], got {keep}")
    if keep == total:
        return list(range(total))
    if keep == 0:
        return []
    indices = np.random.default_rng(seed).choice(total, size=keep, replace=False)
    indices.sort()
    return indices.tolist()
