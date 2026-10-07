# SPDX-License-Identifier: Apache-2.0
"""CPU policy tests for Whisper ASR (no accelerator required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni import platforms
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.cpu import CPUOmniPlatform


def _builder() -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )


def _server_args(attention_backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=None,
        attention_backend=attention_backend,
    )


def test_whisper_cpu_selects_torch_native_attention(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", CPUOmniPlatform())

    defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"


def test_whisper_cpu_rejects_unsupported_attention_backend() -> None:
    with pytest.raises(ValueError, match="requires attention_backend='torch_native'"):
        CPUOmniPlatform().apply_model_worker_backend_policy(
            _server_args("triton"),
            SimpleNamespace(quantization=None),
            "WhisperForConditionalGeneration",
        )


def test_whisper_cpu_accepts_torch_native_attention_backend() -> None:
    effective_quantization = CPUOmniPlatform().apply_model_worker_backend_policy(
        _server_args("torch_native"),
        SimpleNamespace(quantization=None),
        "WhisperForConditionalGeneration",
    )

    assert effective_quantization is None
