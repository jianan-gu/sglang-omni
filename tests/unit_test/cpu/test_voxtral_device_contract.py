# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS device and compile contracts on CPU.

Kept here rather than appended to ``tests/unit_test/test_stage_device_contract.py``
so each model's CPU enablement lands on its own branch without colliding with the
others in one shared file, and so CI can select CPU coverage by directory.

The stage-level "config leaves device unset" half is asserted directly below,
since these stages are not in that shared registry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang_omni.platforms as platforms


def test_the_voxtral_stages_leave_device_to_their_factories() -> None:
    """A config-pinned device would be honored as-is and never retargeted, so it
    has to stay unset for the factory resolution below to matter at all.
    """
    from sglang_omni.models.voxtral_tts.config import EntryClass

    config = EntryClass(model_path="unused")

    for stage_name in ("tts_generation", "vocoder"):
        assert config.stage_named(stage_name).factory.device is None, stage_name


def test_voxtral_generation_stage_forwards_none_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generation stage used to pin ``device="cuda:0"`` in its signature
    default, which the config never overrode, so the literal followed the stage
    onto a CPU host and died at torch.cuda.set_device.

    Patching the base builder's build() also proves it is the builder in play: a
    factory using an unrelated builder would leave this spy untouched.
    """
    from sglang_omni.models.voxtral_tts.pipeline import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_generation_executor("unused", gpu_id=1)

    assert "device" in seen, "the factory did not route through the shared builder"
    assert seen["device"] is None
    assert seen["gpu_id"] == 1


@pytest.mark.parametrize(
    ("device", "gpu_id", "expected"),
    [
        # Placement must not overwrite a device the caller named.
        ("cpu", 2, "cpu"),
        ("cuda:3", None, "cuda:3"),
        # Unset device: the platform names the type, placement the index.
        (None, 0, None),
        (None, None, None),
    ],
)
def test_voxtral_vocoder_never_overwrites_an_explicit_device(
    monkeypatch: pytest.MonkeyPatch, device, gpu_id, expected
) -> None:
    """The worst shape of this bug: the vocoder did not merely default to CUDA,
    it *replaced* whatever the caller passed whenever placement supplied a
    gpu_id (``if gpu_id is not None: device = f"cuda:{gpu_id}"``). An explicit
    cpu stage was therefore retargeted to CUDA and only failed later, inside
    torch.cuda.set_device.

    Drives the real factory with the codec load stubbed out, so a regression in
    the resolution actually fails this test.
    """
    from sglang_omni.models.voxtral_tts.pipeline import stages
    from sglang_omni.utils.device import resolve_device_spec

    seen: dict[str, object] = {}

    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "_load_audio_tokenizer",
        lambda checkpoint_dir, audio_config, dev: seen.setdefault("device", dev),
    )
    monkeypatch.setattr(
        stages,
        "_VoxtralTTSVocoder",
        lambda tokenizer: SimpleNamespace(
            build_scheduler=lambda **kwargs: SimpleNamespace()
        ),
    )

    stages.create_vocoder_executor("unused", device=device, gpu_id=gpu_id)

    assert seen["device"] == (
        expected if expected is not None else resolve_device_spec(None, gpu_id)
    )


def test_voxtral_leaves_torch_compile_to_the_platform() -> None:
    """Inductor plans strides from the meta kernel of
    sgl_kernel.rotary_embedding_cpu, which disagrees with the real kernel
    (expected 4096, got 6144), so the compiled graph trips assert_size_stride
    during warmup. Narrow to Voxtral on purpose — Qwen3-ASR compiles fine on
    CPU — so this must not become a platform-wide claim that CPU cannot compile.
    """
    from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
        VoxtralTtsEngineBuilder,
    )

    defaults = VoxtralTtsEngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["enable_torch_compile"] is not platforms.current_platform.is_cpu()
