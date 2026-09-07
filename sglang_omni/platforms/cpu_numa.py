# SPDX-License-Identifier: Apache-2.0
"""NUMA placement for CPU stage processes.

Accelerators place a stage by device index. CPU cannot — torch rejects any
``cpu:N`` — so placement is the thread and memory binding SGLang derives from
``SGLANG_CPU_OMP_THREADS_BIND``. Left unset, every single-rank stage takes
SGLang's ``cpu_ids_by_node[tp_rank]`` default with ``tp_rank == 0``, so every
stage process binds to node 0: its threads and, through the memory policy that
binding also installs, its allocations.

**The binding unit is the OS process, not the stage.** The mask is exported
before ``proc.start()`` and applies to everything in that process, so stages
sharing a process necessarily share a node. Several models rely on this —
MOSS-TTS and Voxtral run all three of their stages in one ``pipeline`` process,
while Qwen3-Omni gives each of its seven stages a process of its own. A map
keyed by stage name is friendlier to write, so it is accepted and translated,
but a request that would split one process across nodes is rejected rather than
silently resolved.

Placement is decided once in the parent, which is the only place that sees every
process, and is stable for a given launch order.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping

logger = logging.getLogger(__name__)

THREADS_BIND_ENV = "SGLANG_CPU_OMP_THREADS_BIND"
NUMA_MAP_ENV = "SGLANG_OMNI_CPU_NUMA_MAP"


class NumaPlacementConflict(ValueError):
    """Two stages in one process were asked for different NUMA nodes."""


def parse_numa_map(raw: str | None) -> dict[str, int]:
    """Parse ``"thinker=0,talker=1"`` into ``{"thinker": 0, "talker": 1}``."""
    if not raw or not raw.strip():
        return {}
    mapping: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name or not value:
            raise ValueError(
                f"{NUMA_MAP_ENV} entries must look like '<stage>=<node>', got "
                f"{item!r}"
            )
        try:
            node = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{NUMA_MAP_ENV} node for stage {name!r} must be an integer, got "
                f"{value!r}"
            ) from exc
        if node < 0:
            raise ValueError(
                f"{NUMA_MAP_ENV} node for stage {name!r} must be >= 0, got {node}"
            )
        mapping[name] = node
    return mapping


def resolve_process_node(
    stage_names: list[str],
    *,
    node_count: int,
    user_map: Mapping[str, int],
    auto_index: int,
) -> int:
    """Pick the node for one process.

    A user entry for any stage in the process decides it. Entries that disagree
    are a conflict the caller cannot satisfy — the mask binds the process, so
    there is no way to honour both — and saying so beats binding to whichever
    stage happened to come first.
    """
    requested = {
        name: user_map[name] % node_count
        for name in stage_names
        if name in user_map
    }
    if requested:
        chosen = set(requested.values())
        if len(chosen) > 1:
            raise NumaPlacementConflict(
                f"{NUMA_MAP_ENV} asks to split one process across NUMA nodes: "
                + ", ".join(f"{n}->node {v}" for n, v in sorted(requested.items()))
                + ". Stages sharing a process share its binding; give them the "
                "same node, or split them into separate processes first."
            )
        return chosen.pop()

    # Auto: spread processes over nodes. Co-locating costs memory capacity and
    # cores, which are hard limits; spreading costs cross-node shared-memory
    # transfers between stages, which is a slowdown. Prefer the hard limit.
    return auto_index % node_count


class _ParentAssignment:
    """Node per process name, assigned once per parent process.

    ``_patched_spawn_env`` runs in the parent, once per process, in launch
    order, so a counter here yields a stable spread. Keyed by process name so a
    repeated call for the same process returns the same node.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, int] = {}

    def node_for(
        self,
        process_name: str,
        stage_names: list[str],
        *,
        node_count: int,
        user_map: Mapping[str, int],
    ) -> int:
        with self._lock:
            cached = self._nodes.get(process_name)
            if cached is not None:
                return cached
            node = resolve_process_node(
                stage_names,
                node_count=node_count,
                user_map=user_map,
                auto_index=len(self._nodes),
            )
            self._nodes[process_name] = node
            return node

    def reset(self) -> None:
        with self._lock:
            self._nodes.clear()


_assignment = _ParentAssignment()


def reset_assignment() -> None:
    """Drop the recorded placement; for tests and for relaunching a pipeline."""
    _assignment.reset()


def cpu_ids_by_node() -> list[str]:
    """NUMA topology, or an empty list when it cannot be read.

    Unreadable sysfs is not fatal: an unset mask leaves SGLang on the node-0
    default that predates this module.
    """
    try:
        from sglang.srt.utils.numa_utils import get_cpu_ids_by_node

        return list(get_cpu_ids_by_node())
    except Exception as exc:  # pragma: no cover - depends on host sysfs
        logger.warning(
            "Could not read the NUMA topology (%s); leaving %s unset, so stages "
            "keep SGLang's node-0 default.",
            exc,
            THREADS_BIND_ENV,
        )
        return []


def threads_bind_env(
    *,
    process_name: str,
    stage_names: list[str],
    tp_size: int,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the ``SGLANG_CPU_OMP_THREADS_BIND`` override for one process."""
    source_env = env if env is not None else os.environ

    if source_env.get(THREADS_BIND_ENV):
        # An operator pinned the cores explicitly; never second-guess that.
        return {}

    if tp_size > 1:
        # A CPU TP group needs one range per rank and every rank must agree on
        # the same split. This per-process hook cannot coordinate that, so leave
        # the group to SGLang's per-rank default rather than emit a mask that
        # contradicts tp_size.
        return {}

    nodes = cpu_ids_by_node()
    if not nodes:
        return {}

    user_map = parse_numa_map(source_env.get(NUMA_MAP_ENV))
    node = _assignment.node_for(
        process_name,
        stage_names,
        node_count=len(nodes),
        user_map=user_map,
    )
    logger.info(
        "Binding CPU process %s (stages: %s) to NUMA node %d of %d.",
        process_name,
        ", ".join(stage_names) or "<none>",
        node,
        len(nodes),
    )
    return {THREADS_BIND_ENV: str(nodes[node])}


def widen_main_thread_to_its_node() -> int | None:
    """Give this process's non-OMP threads the whole node, not just one core.

    ``torch.ops.sgl_kernel.init_cpu_threads_env`` pins each OMP worker to one
    core from inside an ``omp parallel for``. The main thread participates in
    that team, so it executes one iteration and pins *itself* to a single core
    too — an unintended side effect of the loop, not its purpose. Linux affinity
    is inherited, so every thread created afterwards is stuck on that core as
    well.

    Upstream never notices: its CPU path spends essentially all of its time
    inside OMP regions, whose workers carry their own per-core affinity and are
    unaffected. A pipeline that also runs scheduler, transport and callback
    threads does notice, because those inherit the single-core mask.

    Restoring the mask to the node's full CPU set does not disturb the OMP
    workers — their affinity was set per thread and is not re-derived from the
    parent's.

    Returns the number of CPUs the mask was widened to, or None when there was
    nothing to do (already wide, no topology, or the platform forbids it).
    """
    try:
        current = os.sched_getaffinity(0)
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return None
    if len(current) > 1:
        return None  # never narrowed, or already widened

    pinned = next(iter(current))
    # Read the topology from sysfs, not from get_cpu_ids_by_node(): that helper
    # enumerates only CPUs the *current* process can see, and by this point the
    # process is already pinned to one core, so it would report a single-CPU
    # node and there would be nothing to widen to.
    for spec in _sysfs_node_cpulists():
        cpus = _parse_cpulist(spec)
        if pinned in cpus:
            cpus = _drop_hyperthread_siblings(cpus)
            try:
                os.sched_setaffinity(0, cpus)
            except OSError as exc:  # pragma: no cover - cgroup/permission
                logger.warning(
                    "Could not widen the CPU mask from core %d to its node (%s); "
                    "threads outside OMP regions stay on that one core.",
                    pinned,
                    exc,
                )
                return None
            logger.info(
                "Widened the CPU mask from core %d to its %d-core NUMA node so "
                "non-OMP threads are not confined to a single core.",
                pinned,
                len(cpus),
            )
            return len(cpus)
    return None


def _parse_cpulist(spec: str) -> set[int]:
    """Parse a Linux cpulist such as ``"0-39,240-279"`` or ``"0,1,2"``."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _sysfs_node_cpulists() -> list[str]:
    """Per-node cpulists straight from sysfs, unaffected by this process's mask."""
    import glob

    specs: list[str] = []
    for path in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
        try:
            with open(path) as handle:
                specs.append(handle.read().strip())
        except OSError:  # pragma: no cover - depends on host sysfs
            continue
    return specs


def install_binding_hook() -> bool:
    """Widen the CPU mask the moment SGLang narrows it, not minutes later.

    ``init_cpu_threads_env`` runs deep inside SGLang's bootstrap, before the
    model is built. Everything spawned between that call and our first chance to
    react — weight loading, worker pools, transport threads — inherits the
    single-core mask and cannot be moved afterwards, because affinity is only
    inherited at creation. On one Whisper run that gap was 4.5 minutes and left
    70 threads stranded on one core.

    Wrapping the op closes the gap: the mask is widened before the call returns,
    so no thread is ever created under it. Call this in the stage worker before
    any SGLang bootstrap runs.

    Returns True if the hook was installed, False if it was already present or
    the op is unavailable.
    """
    try:
        import sglang.srt.distributed.bootstrap as bootstrap
    except Exception as exc:  # pragma: no cover - depends on the sglang build
        logger.debug("No SGLang CPU bootstrap to hook (%s).", exc)
        return False

    if getattr(bootstrap, "_omni_widen_hooked", False):
        return False

    original = bootstrap._init_cpu_threads_env

    def _wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        widen_main_thread_to_its_node()
        return result

    bootstrap._init_cpu_threads_env = _wrapped
    bootstrap._omni_widen_hooked = True
    return True


def _drop_hyperthread_siblings(cpus: set[int]) -> set[int]:
    """Keep one logical CPU per physical core.

    Siblings share the core's execution units, so handing both to a compute-bound
    thread pool buys contention rather than throughput — and SGLang has already
    sized its OMP team to the physical core count. Where sysfs does not report
    the topology the set is returned untouched, which is the previous behaviour.
    """
    kept: set[int] = set()
    seen: set[int] = set()
    for cpu in sorted(cpus):
        if cpu in seen:
            continue
        siblings = _thread_siblings(cpu)
        if not siblings:
            kept.add(cpu)
            continue
        # Deterministic pick, so every process in a run agrees on the same set.
        kept.add(min(siblings))
        seen.update(siblings)
    return kept or cpus


def _thread_siblings(cpu: int) -> set[int]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as handle:
            return _parse_cpulist(handle.read().strip())
    except OSError:  # pragma: no cover - depends on host sysfs
        return set()
