"""Check experiment selection separately from the arithmetic reference tests.

These checks catch a particularly misleading failure: requesting one variant
but accidentally running an old binary or a module cached for another variant.
"""

import importlib.util
import platform
import sys
from types import SimpleNamespace

import pytest

from codegemm.inference import cpu_backends, cpu_ops


@pytest.mark.parametrize(
    "requested,canonical,device",
    [("cpu", "cpu-1", "cpu"), ("cuda", "cuda", "cuda")]
    + [(name, name, "cpu") for name in cpu_backends.CPU_BACKENDS],
)
def test_backend_names_are_distinct_from_devices(requested, canonical, device):
    assert cpu_backends.canonical_backend(requested) == canonical
    assert cpu_backends.backend_device_type(requested) == device


def test_cpu_thread_defaults_respect_pi_budget_and_affinity(monkeypatch):
    # Restricting affinity to two available CPUs should also reduce the default
    # pool size. Explicit API callers still control their own PyTorch settings.
    monkeypatch.setattr(cpu_backends.os, "sched_getaffinity", lambda _: {0, 1})
    for backend in ("cpu", "cpu-1", "cpu-2"):
        assert cpu_backends.default_cpu_threads(backend) == 1
    for backend in ("cpu-1-mt", "cpu-3", "cpu-4"):
        assert cpu_backends.default_cpu_threads(backend) == 2
        with pytest.warns(UserWarning, match="capping"):
            assert cpu_backends.resolve_cpu_threads(backend, 8) == 4
    with pytest.warns(UserWarning, match="single threaded"):
        assert cpu_backends.resolve_cpu_threads("cpu-1", 2) == 2
    with pytest.raises(ValueError, match="positive"):
        cpu_backends.resolve_cpu_threads("cpu-1", 0)


def test_old_optimized_baseline_requires_rebuild(monkeypatch):
    # A previously installed -O2 extension must not be silently called cpu-1.
    monkeypatch.setattr(cpu_ops, "_CPU_EXTENSIONS", {})
    monkeypatch.setitem(sys.modules, "codegemm_cpu_kernel", SimpleNamespace())
    with pytest.raises(RuntimeError, match="older build.*metadata"):
        cpu_ops._cpu_extension("cpu-1")


def test_wrong_variant_binary_is_rejected(monkeypatch):
    monkeypatch.setattr(cpu_ops, "_CPU_EXTENSIONS", {})
    fake = SimpleNamespace(build_info=lambda: {"backend": "cpu-4"})
    monkeypatch.setitem(sys.modules, "codegemm_cpu_kernel", fake)
    with pytest.raises(RuntimeError, match="Requested cpu-1.*reports"):
        cpu_ops._cpu_extension("cpu-1")


@pytest.mark.parametrize("backend", cpu_backends.CPU_BACKENDS)
def test_compiled_metadata_matches_requested_experiment(backend):
    if backend in ("cpu-2", "cpu-3", "cpu-4"):
        if platform.machine().lower() not in ("aarch64", "arm64"):
            pytest.skip("Native NEON execution requires AArch64")
    if backend != "cpu-1":
        if importlib.util.find_spec(cpu_backends.cpu_extension_name(backend)) is None:
            pytest.skip(f"Optional extension {backend} has not been built")
    info = cpu_ops.cpu_backend_info(backend)
    neon = backend in ("cpu-2", "cpu-3", "cpu-4")
    threaded = backend in ("cpu-1-mt", "cpu-3", "cpu-4")
    optimized = backend == "cpu-4"
    assert info["backend"] == backend
    assert info["neon"] is neon
    assert info["multithreaded"] is threaded
    assert info["optimized"] is optimized
    assert info["optimization_level"] == ("O3" if optimized else "O0")
    assert info["max_threads"] == (4 if threaded else 1)
    assert "-ffast-math" not in info["compiler_flags"].split()
    assert ("-O3" if optimized else "-O0") in info["compiler_flags"].split()
