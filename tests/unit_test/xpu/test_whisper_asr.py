# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for Whisper ASR."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni import platforms
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.xpu import XPUOmniPlatform


def test_whisper_selects_torch_native_attention(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", XPUOmniPlatform())
    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )
    defaults = builder.generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"


def test_whisper_policy_does_not_reject_intel_xpu_override() -> None:
    server_args = SimpleNamespace(
        quantization=None,
        moe_runner_backend="auto",
        attention_backend="intel_xpu",
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    effective_quantization = XPUOmniPlatform().apply_model_worker_backend_policy(
        server_args,
        SimpleNamespace(quantization=None),
        "WhisperForConditionalGeneration",
    )

    assert effective_quantization is None
