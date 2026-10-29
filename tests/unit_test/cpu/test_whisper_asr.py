# SPDX-License-Identifier: Apache-2.0
"""CPU policy tests for Whisper ASR (no accelerator required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni import platforms
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.scheduling import sglang_backend


def whisper_builder() -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        encoder_attention_backend="auto",
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )


def server_args(
    attention_backend: str,
    prefill_attention_backend: str | None = None,
    decode_attention_backend: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=None,
        attention_backend=attention_backend,
        prefill_attention_backend=prefill_attention_backend,
        decode_attention_backend=decode_attention_backend,
    )


def test_whisper_cpu_selects_torch_native_attention(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", CPUOmniPlatform())

    defaults = whisper_builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["enable_torch_compile"] is False


def test_whisper_accelerator_keeps_compile_and_async_decode(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", CUDAOmniPlatform())
    builder = whisper_builder()
    builder.device = "cuda:0"

    assert builder.generation_defaults(dtype="float16")["enable_torch_compile"] is True
    assert builder.extra_scheduler_kwargs()["enable_async_decode"] is True


def test_explicit_cpu_stage_on_accelerator_host_uses_cpu_defaults(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", CUDAOmniPlatform())
    builder = whisper_builder()
    builder.context_length = 1600
    monkeypatch.setattr(builder, "pre_infra_setup", lambda checkpoint_dir: None)
    captured: dict[str, object] = {}

    class StopBeforeModelLoad(Exception):
        pass

    def capture_server_args(model_path: str, **overrides: object) -> None:
        captured.update(overrides)
        raise StopBeforeModelLoad

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", capture_server_args)

    with pytest.raises(StopBeforeModelLoad):
        builder.build("unused", device="cpu", dtype="float16")

    assert captured["device"] == "cpu"
    assert captured["attention_backend"] == "torch_native"
    assert captured["disable_cuda_graph"] is True
    assert captured["enable_torch_compile"] is False


def test_whisper_cpu_disables_async_decode_even_when_requested() -> None:
    builder = whisper_builder()
    builder.device = "cpu:0"

    assert builder.extra_scheduler_kwargs()["enable_async_decode"] is False


@pytest.mark.parametrize(
    "attention_args",
    [
        server_args("triton"),
        server_args("torch_native", prefill_attention_backend="triton"),
        server_args("torch_native", decode_attention_backend="triton"),
    ],
)
def test_whisper_cpu_rejects_unsupported_attention_backend(attention_args) -> None:
    with pytest.raises(ValueError, match="requires torch_native"):
        CPUOmniPlatform().apply_model_worker_backend_policy(
            attention_args,
            SimpleNamespace(quantization=None),
            "WhisperForConditionalGeneration",
        )


def test_whisper_cpu_accepts_torch_native_attention_backend() -> None:
    effective_quantization = CPUOmniPlatform().apply_model_worker_backend_policy(
        server_args("torch_native"),
        SimpleNamespace(quantization=None),
        "WhisperForConditionalGeneration",
    )

    assert effective_quantization is None
