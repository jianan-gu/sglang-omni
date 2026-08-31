# SPDX-License-Identifier: Apache-2.0
"""CPU worker processes must be spread across NUMA nodes.

Accelerators place a stage by device index. CPU cannot — torch rejects any
``cpu:N`` — so placement is the thread and memory binding SGLang derives from
``SGLANG_CPU_OMP_THREADS_BIND``. Without it every single-rank stage takes
SGLang's ``cpu_ids_by_node[tp_rank]`` default with ``tp_rank == 0``, so every
process binds to node 0 while each sizes its memory pool as if it owned a whole
node. That is the shape that OOM-killed Qwen3-Omni with ~1.4 TB free.

The binding unit is the **process**: the mask is exported before
``proc.start()`` and covers every thread and allocation in it. Stages sharing a
process therefore share a node, which several models rely on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.platforms import cpu_numa
from sglang_omni.platforms.cpu import CPUOmniPlatform

_BIND = cpu_numa.THREADS_BIND_ENV
_MAP = cpu_numa.NUMA_MAP_ENV


@pytest.fixture(autouse=True)
def fresh_assignment():
    """Placement is recorded per parent process; tests must not inherit it."""
    cpu_numa.reset_assignment()
    yield
    cpu_numa.reset_assignment()


@pytest.fixture
def nodes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fixed four-node topology, so tests do not depend on the host."""
    topology = ["0-9", "10-19", "20-29", "30-39"]
    monkeypatch.setattr(cpu_numa, "cpu_ids_by_node", lambda: list(topology))
    return topology


def _stage(name: str, tp_size: int = 1):
    return SimpleNamespace(stage_name=name, tp_size=tp_size, gpu_id=0)


def _bind(platform, process_name, stage_names, env, tp_size: int = 1):
    specs = [_stage(n, tp_size) for n in stage_names]
    return platform.get_process_placement_env(process_name, specs, env=env)


# --- automatic spreading ------------------------------------------------


def test_separate_processes_land_on_separate_nodes(nodes) -> None:
    """The whole point: two processes must not share a node, or they each
    budget a full node's memory against the same node.
    """
    platform = CPUOmniPlatform()

    masks = [
        _bind(platform, f"proc{i}", [f"stage{i}"], env={})[_BIND]
        for i in range(len(nodes))
    ]

    assert masks == nodes
    assert len(set(masks)) == len(nodes)


def test_more_processes_than_nodes_wrap(nodes) -> None:
    """A pipeline wider than the host must still start; the extra processes
    share nodes again, which is the pre-existing behaviour, not a new failure.
    """
    platform = CPUOmniPlatform()

    masks = [
        _bind(platform, f"proc{i}", [f"stage{i}"], env={})[_BIND]
        for i in range(len(nodes) + 2)
    ]

    assert masks[len(nodes) :] == nodes[:2]


def test_stages_sharing_a_process_share_its_node(nodes) -> None:
    """MOSS-TTS and Voxtral run all three stages in one process; the binding
    covers the process, so they cannot be split and must not be double-counted
    as separate placements.
    """
    platform = CPUOmniPlatform()

    first = _bind(platform, "pipeline", ["pre", "engine", "vocoder"], env={})
    second = _bind(platform, "other", ["extra"], env={})

    assert first[_BIND] == nodes[0]
    assert second[_BIND] == nodes[1]


def test_asking_twice_for_one_process_is_stable(nodes) -> None:
    platform = CPUOmniPlatform()

    first = _bind(platform, "pipeline", ["a"], env={})
    second = _bind(platform, "pipeline", ["a"], env={})

    assert first == second


# --- user-supplied map --------------------------------------------------


def test_a_user_map_places_the_named_stage(nodes) -> None:
    platform = CPUOmniPlatform()

    mask = _bind(platform, "p", ["talker"], env={_MAP: "talker=2"})[_BIND]

    assert mask == nodes[2]


def test_a_user_map_entry_wraps_like_the_automatic_one(nodes) -> None:
    platform = CPUOmniPlatform()

    mask = _bind(platform, "p", ["talker"], env={_MAP: f"talker={len(nodes)}"})[_BIND]

    assert mask == nodes[0]


def test_unnamed_processes_still_spread_around_a_user_entry(nodes) -> None:
    """A partial map must not collapse the rest onto one node."""
    platform = CPUOmniPlatform()

    pinned = _bind(platform, "b", ["talker"], env={_MAP: "talker=3"})[_BIND]
    auto = _bind(platform, "a", ["thinker"], env={_MAP: "talker=3"})[_BIND]

    assert pinned == nodes[3]
    assert auto != pinned


def test_splitting_one_process_across_nodes_is_refused(nodes) -> None:
    """The mask binds the process, so honouring both entries is impossible.
    Saying so beats silently binding to whichever stage came first.
    """
    platform = CPUOmniPlatform()

    with pytest.raises(cpu_numa.NumaPlacementConflict, match="split one process"):
        _bind(platform, "pipeline", ["pre", "vocoder"], env={_MAP: "pre=0,vocoder=2"})


def test_agreeing_entries_in_one_process_are_fine(nodes) -> None:
    platform = CPUOmniPlatform()

    mask = _bind(platform, "pipeline", ["pre", "vocoder"], env={_MAP: "pre=2,vocoder=2"})

    assert mask[_BIND] == nodes[2]


@pytest.mark.parametrize("raw", ["thinker", "thinker=x", "thinker=-1", "=0"])
def test_a_malformed_map_is_rejected_with_its_own_text(raw: str) -> None:
    with pytest.raises(ValueError, match=_MAP):
        cpu_numa.parse_numa_map(raw)


def test_an_empty_map_is_not_an_error() -> None:
    assert cpu_numa.parse_numa_map(None) == {}
    assert cpu_numa.parse_numa_map("  ") == {}


# --- deferrals ----------------------------------------------------------


def test_an_operator_supplied_mask_is_never_second_guessed(nodes) -> None:
    """Overriding the binding is how someone tunes a deployment; replacing
    their mask would make the knob useless.
    """
    platform = CPUOmniPlatform()

    assert _bind(platform, "p", ["s"], env={_BIND: "0-3"}) == {}


def test_a_tp_group_is_left_to_sglang(nodes) -> None:
    """A CPU TP group needs one range per rank and all ranks must agree on the
    split. This per-process hook cannot coordinate that, so it must not emit a
    mask that contradicts tp_size.
    """
    platform = CPUOmniPlatform()

    assert _bind(platform, "p", ["s"], env={}, tp_size=4) == {}


def test_an_unreadable_topology_degrades_to_the_previous_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing to read sysfs must not break startup — it leaves SGLang on the
    node-0 default that predates this hook.
    """
    monkeypatch.setattr(cpu_numa, "cpu_ids_by_node", lambda: [])

    assert CPUOmniPlatform().get_process_placement_env("p", [_stage("s")], env={}) == {}


def test_a_process_with_no_stages_is_left_alone(nodes) -> None:
    assert CPUOmniPlatform().get_process_placement_env("p", [], env={}) == {}


# --- wiring -------------------------------------------------------------


def test_every_process_asks_the_platform_not_just_tp_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spawn-time hook used to short-circuit unless a stage had tp_size > 1.

    That suits accelerators, which place by device index and need no env below
    TP, but it left CPU — whose only placement channel *is* env — unable to say
    anything at all.
    """
    from sglang_omni.pipeline import stage_workers

    asked: list[tuple[str, list[str]]] = []

    class _Platform:
        def get_stage_process_env(self, spec, env=None):
            raise AssertionError("the per-stage hook is for TP stages only")

        def get_process_placement_env(self, process_name, stage_specs, env=None):
            asked.append((process_name, [s.stage_name for s in stage_specs]))
            return {"MARKER": process_name}

    monkeypatch.setattr(stage_workers, "current_platform", _Platform())

    spec = SimpleNamespace(
        process_name="pipeline",
        stage_specs=[_stage("pre"), _stage("vocoder")],
    )

    assert stage_workers._get_worker_process_env(spec) == {"MARKER": "pipeline"}
    assert asked == [("pipeline", ["pre", "vocoder"])]


def test_a_tp_stage_still_must_own_its_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relaxing the gate must not relax the invariant behind it: a TP stage's
    device remap and NCCL settings assume it is the sole tenant.
    """
    from sglang_omni.pipeline import stage_workers

    monkeypatch.setattr(
        stage_workers,
        "current_platform",
        SimpleNamespace(
            get_stage_process_env=lambda spec, env=None: {},
            get_process_placement_env=lambda p, s, env=None: {},
        ),
    )

    spec = SimpleNamespace(
        process_name="pipeline",
        stage_specs=[_stage("thinker", tp_size=2), _stage("decode")],
    )

    with pytest.raises(AssertionError, match="must own their OS process"):
        stage_workers._get_worker_process_env(spec)


def test_accelerators_are_unaffected() -> None:
    """The new hook defaults to {} so CUDA and XPU behaviour cannot shift."""
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    for cls in (CUDAOmniPlatform, XPUOmniPlatform):
        platform = cls.__new__(cls)
        assert platform.get_process_placement_env("p", [_stage("s")], env={}) == {}
