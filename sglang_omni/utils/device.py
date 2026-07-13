# SPDX-License-Identifier: Apache-2.0
"""Accelerator-agnostic device helpers.

Centralizes `cuda`/`xpu`/... dispatch so callers stop hardcoding
``torch.cuda.*`` calls and ``"cuda:{id}"`` device strings. Mirrors the
dispatch-key concept from ``sglang.srt.platforms`` (``current_platform``)
but stays dependency-free so it can be used from code paths (relay
transports, CLI) that must not pull in the full sglang runtime.
"""

from __future__ import annotations

import torch

_VISIBLE_DEVICES_ENV_VAR = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}

# torch.distributed backend used for cross-process collectives, per device type.
_DIST_BACKEND = {
    "cuda": "nccl",
    "xpu": "ccl",
}


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is None


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def current_accelerator_type() -> str:
    """Return the accelerator device type visible to this process.

    Checked in a fixed order (cuda, then xpu) so a host with a single
    accelerator kind resolves unambiguously. Falls back to "cpu".
    """
    if is_cuda_available():
        return "cuda"
    if is_xpu_available():
        return "xpu"
    return "cpu"


def get_device_module(device_type: str | None = None):
    """Return the ``torch.<device_type>`` submodule, or None for cpu."""
    device_type = device_type or current_accelerator_type()
    if device_type == "cpu":
        return None
    get_module = getattr(torch, "get_device_module", None)
    if get_module is not None:
        try:
            return get_module(device_type)
        except Exception:
            pass
    return getattr(torch, device_type, None)


def resolve_device(logical_gpu_id: int, device_type: str | None = None) -> torch.device:
    """Build a ``torch.device`` for a logical accelerator id without hardcoding cuda."""
    device_type = device_type or current_accelerator_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(f"{device_type}:{logical_gpu_id}")


def set_device(device: torch.device) -> None:
    """Set the active device for the calling thread, if the device type supports it."""
    if device.type == "cpu":
        return
    module = get_device_module(device.type)
    if module is not None and hasattr(module, "set_device"):
        module.set_device(device)


def new_event(device_type: str | None = None, **kwargs):
    """Create a device event (``torch.cuda.Event``, ``torch.xpu.Event``, ...)."""
    module = get_device_module(device_type)
    if module is None or not hasattr(module, "Event"):
        raise RuntimeError(f"No event type available for device_type={device_type!r}")
    return module.Event(**kwargs)


def synchronize(device: torch.device | str | None = None) -> None:
    """Synchronize the given device (or the current accelerator). No-op on cpu."""
    device_type = (
        device.type if isinstance(device, torch.device) else device
    ) or current_accelerator_type()
    if device_type == "cpu":
        return
    module = get_device_module(device_type)
    if module is not None and hasattr(module, "synchronize"):
        module.synchronize()


def device_count(device_type: str | None = None) -> int:
    module = get_device_module(device_type)
    if module is None or not hasattr(module, "device_count"):
        return 0
    return int(module.device_count())


def current_device_index(device_type: str | None = None) -> int:
    module = get_device_module(device_type)
    if module is None or not hasattr(module, "current_device"):
        return 0
    return int(module.current_device())


def visible_devices_env_var(device_type: str | None = None) -> str:
    """Env var used to restrict which accelerators of this type are visible."""
    device_type = device_type or current_accelerator_type()
    return _VISIBLE_DEVICES_ENV_VAR.get(device_type, "CUDA_VISIBLE_DEVICES")


def dist_backend(device_type: str | None = None) -> str:
    """``torch.distributed`` backend to use for collectives on this device type."""
    device_type = device_type or current_accelerator_type()
    return _DIST_BACKEND.get(device_type, "nccl")
