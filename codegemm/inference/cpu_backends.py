"""Names and device rules shared by the CPU experiments and their entry points.

The backend selects a CodeGEMM implementation; the device selects a physical
execution location. ``cpu-2`` is therefore a backend name, never a torch.device.
This module intentionally imports neither PyTorch nor a compiled extension.
"""

import os
import platform
import warnings


CPU_BACKENDS = ("cpu-1", "cpu-2", "cpu-3", "cpu-4")
BACKEND_CHOICES = ("cuda", "cpu", *CPU_BACKENDS)
_EXTENSION_NAMES = {
    "cpu-1": "codegemm_cpu_kernel",
    "cpu-2": "codegemm_cpu_kernel_2",
    "cpu-3": "codegemm_cpu_kernel_3",
    "cpu-4": "codegemm_cpu_kernel_4",
}


def canonical_backend(backend):
    """Keep old ``cpu`` commands working by treating them as ``cpu-1``."""
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unknown CodeGEMM backend: {backend!r}")
    return "cpu-1" if backend == "cpu" else backend


def is_cpu_backend(backend):
    return canonical_backend(backend) in CPU_BACKENDS


def backend_device_type(backend):
    """All four CPU implementations use ordinary CPU tensors."""
    return "cpu" if is_cpu_backend(backend) else "cuda"


def cpu_extension_name(backend):
    backend = canonical_backend(backend)
    if backend not in _EXTENSION_NAMES:
        raise ValueError("A CUDA backend has no CPU extension")
    return _EXTENSION_NAMES[backend]


def validate_backend_platform(backend):
    """Never pretend that a scalar x86 fallback is an ARM NEON experiment."""
    backend = canonical_backend(backend)
    if backend in ("cpu-2", "cpu-3", "cpu-4"):
        machine = platform.machine().lower()
        if machine not in ("aarch64", "arm64"):
            raise RuntimeError(
                f"{backend} requires a 64-bit ARM CPU with NEON; this process "
                f"is running on {machine}. Use cpu-1 on x86/WSL and run the "
                "NEON variants on the Raspberry Pi."
            )


def default_cpu_threads(backend):
    """Use one core for early variants and at most the Pi 5's four cores later."""
    backend = canonical_backend(backend)
    if backend not in CPU_BACKENDS:
        raise ValueError("CPU thread defaults require a CPU backend")
    if backend in ("cpu-1", "cpu-2"):
        return 1
    # Affinity accounts for containers or a taskset restriction. Fall back to
    # logical CPU count on platforms that do not expose sched_getaffinity().
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return min(4, max(1, available))


def resolve_cpu_threads(backend, requested=None):
    """Choose CLI thread settings without changing any global PyTorch state.

    Entrypoints apply the returned count explicitly. The loading API does not
    call this function, so creating a layer never changes another model's pool.
    """
    backend = canonical_backend(backend)
    if backend not in CPU_BACKENDS:
        raise ValueError("CPU thread settings require a CPU backend")
    if requested is None:
        return default_cpu_threads(backend)
    if requested < 1:
        raise ValueError("CPU thread count must be positive")
    if backend in ("cpu-3", "cpu-4") and requested > 4:
        warnings.warn(
            f"{backend} is designed for the Pi 5's four cores; capping the "
            "custom kernels and framework CPU thread setting at 4.",
            stacklevel=2,
        )
        return 4
    if backend in ("cpu-1", "cpu-2") and requested > 1:
        warnings.warn(
            f"{backend} custom kernels remain single threaded. Extra threads "
            "can accelerate framework matrix multiplication/attention; use 1 "
            "for the intended cumulative-variant comparison.",
            stacklevel=2,
        )
    return requested
