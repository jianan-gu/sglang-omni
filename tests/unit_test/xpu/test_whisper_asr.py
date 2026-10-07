# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for Whisper ASR."""

from __future__ import annotations

from sglang_omni import platforms
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
)


def whisper_builder() -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )


def test_whisper_selects_torch_native_attention(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", XPUOmniPlatform())

    defaults = whisper_builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"


def test_whisper_explicit_attention_backend_overrides_platform_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(platforms, "current_platform", XPUOmniPlatform())

    overrides = build_generation_batch_overrides(
        server_args_overrides={
            "attention_backend": "intel_xpu",
            "disable_cuda_graph": True,
        },
        **whisper_builder().generation_defaults(dtype="float16"),
    )

    assert overrides["attention_backend"] == "intel_xpu"
    # intel_xpu needs every graph off for encoder-decoder models, so the
    # disable flag must also turn off the stage-default breakable prefill graph.
    assert overrides["cuda_graph_backend_prefill"] == CudaGraphBackend.DISABLED
