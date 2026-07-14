# SPDX-License-Identifier: Apache-2.0
"""Template builder for TTS SGLang AR engine stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.utils.device import current_accelerator_type, resolve_device

logger = logging.getLogger(__name__)


class TtsEngineBuilder(ABC):
    """Base builder for TTS AR engine stages."""

    model_name: str
    context_length: int
    model_arch_override: str | None = None

    def build(
        self,
        model_path: str,
        *,
        device: str | None = None,
        gpu_id: int | None = None,
        dtype: str = "bfloat16",
        server_args_overrides: dict[str, Any] | None = None,
    ) -> Any:
        from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
        from sglang_omni.scheduling import sglang_backend

        checkpoint_dir = self.resolve_checkpoint(model_path)
        accel_type = current_accelerator_type()
        if gpu_id is not None:
            device = str(resolve_device(gpu_id, accel_type))
        elif device is None:
            device = str(resolve_device(0, accel_type))
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype

        self.pre_infra_setup(checkpoint_dir)

        overrides = build_generation_batch_overrides(
            server_args_overrides=server_args_overrides,
            **self.generation_defaults(dtype=dtype),
        )
        self.adjust_overrides(overrides)

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)

        if bool(getattr(server_args, "enable_torch_compile", False)) and accel_type != "cuda":
            # Must happen before create_sglang_infrastructure_defer_cuda_graph:
            # ModelRunner construction calls RuntimeContext.set_server_args(),
            # which snapshots enable_torch_compile into a process-global flag
            # at that moment (sglang.srt.runtime_context.set_server_args) —
            # mutating server_args after that point would not affect the
            # already-captured snapshot. torch.compile's max-autotune search
            # (this repo's default compile mode, see
            # qwen3_tts._compile_qwen3_tts_backbone) autotunes on a much less
            # mature Inductor/Triton backend on non-CUDA devices and has been
            # observed to stall for hours with no error and no log output.
            logger.info(
                "Disabling torch.compile: no CUDA accelerator detected (device=%s)",
                accel_type,
            )
            server_args.enable_torch_compile = False

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
        want_cuda_graph, (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            prefill_mgr,
            decode_mgr,
            model_config,
        ) = scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            gpu_id,
            **infra_kwargs,
        )
        model = model_worker.model_runner.model

        self.setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )

        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
            model_buffer_bs=self.get_model_buffer_bs(model),
        )

        self.compile_model(model, server_args)

        if want_cuda_graph:
            model_worker.model_runner.init_device_graphs()
            self.post_cuda_graph_setup(model, server_args)

        output_proc = sglang_backend.SGLangOutputProcessor(
            capture_hidden=False,
            capture_hidden_layers=None,
            model=model,
        )
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)

        scheduler = self.make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_mgr,
            decode_manager=decode_mgr,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        self.post_scheduler_setup(scheduler, model_runner)
        return scheduler

    @abstractmethod
    def resolve_checkpoint(self, model_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        del overrides

    def customize_server_args(self, server_args: Any) -> None:
        del server_args

    def infra_kwargs(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        raise NotImplementedError

    def get_model_buffer_bs(self, model: Any) -> int | None:
        del model
        return None

    def compile_model(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def make_abort_callback(self) -> Any | None:
        return None

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {}

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            **self.extra_scheduler_kwargs(),
        )

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        del scheduler, model_runner
