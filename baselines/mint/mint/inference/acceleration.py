"""Optional inference acceleration for fixed-shape CUDA workloads."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass


COMPILE_MODES = (
    "auto",
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
FP8_MODES = ("auto", "dynamic")
_CUDAGRAPH_MODES = {"auto", "reduce-overhead", "max-autotune"}
_DISABLED_VALUES = {"", "0", "false", "none", "off"}


def normalize_optional_mode(value, *, choices: tuple[str, ...], name: str) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        value = choices[0]
    normalized = str(value).strip().lower()
    if normalized in _DISABLED_VALUES:
        return None
    if normalized not in choices:
        raise ValueError(f"Unsupported {name}={value!r}; choose from: {', '.join(choices)}")
    return normalized


def prepare_allocator_for_compile(compile_mode: str | None) -> bool:
    """Disable expandable segments before CUDA initialization for CUDA graphs."""
    if compile_mode not in _CUDAGRAPH_MODES:
        return False

    changed = False
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        raw = os.environ.get(key)
        if not raw:
            continue
        entries = [item.strip() for item in raw.split(",") if item.strip()]
        kept = [
            item for item in entries
            if item.lower().replace(" ", "") != "expandable_segments:true"
        ]
        if len(kept) == len(entries):
            continue

        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            raise RuntimeError(
                "CUDA is already initialized with expandable_segments enabled; "
                "construct StudentEngine with compile_mode before the first CUDA operation"
            )
        changed = True
        if kept:
            os.environ[key] = ",".join(kept)
        else:
            os.environ.pop(key, None)
    return changed


def resolve_compile_mode(requested: str | None, devices) -> tuple[str | None, str]:
    if requested != "auto":
        return requested, "explicit"
    if any(device.type == "cuda" for device in devices):
        return "reduce-overhead", "auto CUDA"
    return None, "auto disabled without CUDA"


def resolve_fp8_mode(
    requested: str | None,
    devices,
    compile_mode: str | None,
) -> tuple[str | None, str]:
    if requested != "auto":
        return requested, "explicit"
    if compile_mode is None:
        return None, "auto FP8 requires torch.compile"
    cuda_devices = [device for device in devices if device.type == "cuda"]
    if len(cuda_devices) != len(devices) or not cuda_devices:
        return None, "auto FP8 requires CUDA-only devices"
    if importlib.util.find_spec("torchao") is None:
        return None, "torchao is unavailable"

    import torch

    unsupported = [
        str(device)
        for device in cuda_devices
        if torch.cuda.get_device_capability(device) < (8, 9)
    ]
    if unsupported:
        return None, f"FP8 unsupported on {','.join(unsupported)}"
    return "dynamic", "auto CUDA capability >= 8.9"


def _eligible_fp8_linear(module, fqn: str) -> bool:
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    if not fqn.startswith("backbone.aggregator."):
        return False
    return (
        module.in_features >= 256
        and module.out_features >= 256
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


@dataclass(frozen=True)
class FP8Conversion:
    module_names: tuple[str, ...]
    torchao_version: str


def apply_dynamic_fp8(model, device) -> FP8Conversion:
    """Quantize large aggregator Linear layers while keeping output heads in BF16."""
    import torch

    if device.type != "cuda":
        raise RuntimeError("FP8 inference requires a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        raise RuntimeError(
            f"FP8 inference requires CUDA capability >= 8.9, got {capability} on {device}"
        )
    try:
        import torchao
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )
    except Exception as exc:
        raise RuntimeError(
            "FP8 mode requires a torchao release compatible with the installed PyTorch"
        ) from exc

    eligible = [
        (name, module)
        for name, module in model.named_modules()
        if _eligible_fp8_linear(module, name)
    ]
    if not eligible:
        raise RuntimeError("No eligible aggregator Linear layers were found for FP8")

    for _name, module in eligible:
        module.to(dtype=torch.bfloat16)

    eligible_ids = {id(module) for _name, module in eligible}
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(),
        filter_fn=lambda module, _fqn: id(module) in eligible_ids,
    )
    return FP8Conversion(
        module_names=tuple(name for name, _module in eligible),
        torchao_version=str(getattr(torchao, "__version__", "unknown")),
    )


def compile_hotspots(model, compile_mode: str) -> tuple[str, ...]:
    """Compile fixed-shape blocks without tracing the stateful model loop."""
    import torch

    class FixedShapeCompiledModule(torch.nn.Module):
        """Use the captured shape and keep uncommon tail shapes in eager mode."""

        def __init__(self, module):
            super().__init__()
            object.__setattr__(self, "_eager_module", module)
            self.compiled_module = torch.compile(
                module, mode=compile_mode, fullgraph=False, dynamic=False
            )
            self._compiled_shape = None

        def forward(self, *args, **kwargs):
            first = args[0] if args else None
            shape = tuple(first.shape) if torch.is_tensor(first) else None
            if self._compiled_shape is None:
                self._compiled_shape = shape
            if shape != self._compiled_shape:
                return self._eager_module(*args, **kwargs)
            return self.compiled_module(*args, **kwargs)

    def fixed_shape_compile(module):
        return FixedShapeCompiledModule(module)

    aggregator = model.backbone.aggregator
    compiled = []
    for index, block in enumerate(aggregator.frame_blocks):
        aggregator.frame_blocks[index] = fixed_shape_compile(block)
        compiled.append(f"backbone.aggregator.frame_blocks.{index}")

    patch_blocks = getattr(aggregator.patch_embed, "blocks", None)
    if patch_blocks is not None:
        for index, block in enumerate(patch_blocks):
            patch_blocks[index] = fixed_shape_compile(block)
            compiled.append(f"backbone.aggregator.patch_embed.blocks.{index}")

    # SDPABlock receives a changing global_idx, so tracing the complete global
    # block causes repeated recompilation in fixed-window inference.
    if not getattr(aggregator, "use_sdpa", False):
        for index, block in enumerate(aggregator.global_blocks):
            if hasattr(block, "attn_pre"):
                block.attn_pre = fixed_shape_compile(block.attn_pre)
                compiled.append(f"backbone.aggregator.global_blocks.{index}.attn_pre")
            if hasattr(block, "ffn_residual"):
                block.ffn_residual = fixed_shape_compile(block.ffn_residual)
                compiled.append(f"backbone.aggregator.global_blocks.{index}.ffn_residual")
            block.attn.proj = fixed_shape_compile(block.attn.proj)
            compiled.append(f"backbone.aggregator.global_blocks.{index}.attn.proj")
    return tuple(compiled)
