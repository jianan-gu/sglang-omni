from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms import cpu_numa
from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self):
        return False

    def supports_generation_cuda_graph(self) -> bool:
        return False

    def prepare_worker_process(self) -> None:
        """Arm the widen-on-bind hook before SGLang can narrow the mask."""
        cpu_numa.install_binding_hook()

    def set_device(self, device) -> None:
        """CPU has no device to select; used as the post-binding hook.

        This runs in the stage worker once SGLang has already called
        ``init_cpu_threads_env``, which is the only point where the single-core
        mask it leaves behind can be undone. Idempotent — see
        ``cpu_numa.widen_main_thread_to_its_node``.
        """
        super().set_device(device)
        cpu_numa.widen_main_thread_to_its_node()

    def get_process_placement_env(
        self,
        process_name: str,
        stage_specs: list[StageLaunchConfig],
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Bind this worker process to one NUMA node.

        See :mod:`sglang_omni.platforms.cpu_numa` for why the unit is the
        process and what happens when stages disagree about the node.
        """
        if not stage_specs:
            return {}
        return cpu_numa.threads_bind_env(
            process_name=process_name,
            stage_names=[spec.stage_name for spec in stage_specs],
            tp_size=max(spec.tp_size for spec in stage_specs),
            env=env,
        )
