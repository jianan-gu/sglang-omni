# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for MOSS-TTS Local."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang_omni import platforms
from sglang_omni.models.moss_tts_local import engine_builder, stages
from sglang_omni.models.moss_tts_local import streaming_vocoder as vocoder_module
from sglang_omni.models.moss_tts_local.streaming_vocoder import _xpu_codec_autocast


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)


def test_moss_tts_local_codec_device_follows_xpu_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    assert stages._resolve_codec_device(None, 2) == "xpu:2"
    assert stages._resolve_codec_device("cpu", 2) == "cpu"
    assert stages._resolve_codec_device("xpu", 2) == "xpu:2"
    assert stages._resolve_codec_device("xpu:0", 2) == "xpu:0"


def test_moss_tts_local_xpu_defaults_use_platform_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeXPU:
        @staticmethod
        def device_count() -> int:
            return 2

    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(
        stages.torch,
        "get_device_module",
        lambda device_type: (
            _FakeXPU
            if device_type == "xpu"
            else pytest.fail(f"unexpected device type: {device_type}")
        ),
    )
    monkeypatch.setattr(
        stages.torch.cuda,
        "device_count",
        lambda: pytest.fail("XPU policy must not query torch.cuda"),
    )
    builder = engine_builder.MossTtsLocalEngineBuilder(
        enable_async_decode=False,
        async_decode_min_batch_size=2,
        total_gpu_memory_fraction=None,
        codec_mem_reserve=0.0,
    )

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is True
    assert defaults["sampling_backend"] == "pytorch"
    assert defaults["mem_fraction_static"] == pytest.approx(0.6)


def test_xpu_codec_autocast_uses_codec_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    calls: list[tuple[str, torch.dtype]] = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        yield

    codec = SimpleNamespace(
        compute_dtype=torch.bfloat16,
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="xpu"))]),
    )
    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    with _xpu_codec_autocast(codec):
        pass

    assert calls == [("xpu", torch.bfloat16)]
