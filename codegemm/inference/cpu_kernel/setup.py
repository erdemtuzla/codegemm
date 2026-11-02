"""Build separate binaries so a runtime backend selects an actual experiment.

The optimization flag belongs to each extension, not to a Python switch: a
single binary cannot turn compiler optimization on and off after compilation.
PyTorch itself remains its normal optimized build in all experiments.
"""

import os
import platform
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent
# Keep source paths relative for wheel/sdist portability. This also makes
# direct invocation from the repository root behave like pip's build process.
os.chdir(HERE)
IS_AARCH64 = platform.machine().lower() in ("aarch64", "arm64")
DEFAULT_VARIANTS = "1,1-mt,2,3,4" if IS_AARCH64 else "1,1-mt"
requested = os.environ.get("CODEGEMM_CPU_VARIANTS", DEFAULT_VARIANTS)

# Selecting variants explicitly is useful on a Pi with limited build memory.
# Accept both the CLI spelling (cpu-2) and its shorter build spelling (2).
variants = []
for value in requested.split(","):
    value = value.strip().removeprefix("cpu-")
    if value not in ("1", "1-mt", "2", "3", "4"):
        raise RuntimeError("CODEGEMM_CPU_VARIANTS accepts cpu-1, cpu-1-mt, cpu-2, cpu-3, cpu-4")
    if value not in variants:
        variants.append(value)

if any(variant in ("2", "3", "4") for variant in variants) and not IS_AARCH64:
    raise RuntimeError(
        "cpu-2/cpu-3/cpu-4 contain ARM NEON instructions and require AArch64. "
        "Build cpu-1/cpu-1-mt on this host; build the NEON variants on the 64-bit Pi."
    )

parallel_info = torch.__config__.parallel_info()
uses_openmp = "ATen parallel backend: OpenMP" in parallel_info
uses_native = "ATen parallel backend: native thread pool" in parallel_info
if not uses_native:
    # Some PyTorch versions describe the native pool with a shorter label.
    uses_native = "ATen parallel backend: native" in parallel_info

extensions = []
for selected in variants:
    # cpu-1-mt keeps scalar variant 1's arithmetic/optimization identity, with
    # an additional macro identifying its separate multithreaded module.
    scalar_mt = selected == "1-mt"
    variant = 1 if scalar_mt else int(selected)
    threaded = scalar_mt or variant >= 3
    if threaded and not (uses_openmp or uses_native):
        raise RuntimeError(
            "Threaded CPU variants require an OpenMP or native-thread-pool PyTorch build. "
            "Refusing to build a backend that would silently execute serially."
        )

    # These flags appear after setuptools' default -O3 in the actual compile
    # command. The final -O option wins, making variants 1-3 genuinely -O0.
    # Explicit NEON intrinsics still emit SIMD instructions at -O0.
    # Hide shared C++ helper symbols so loading several modules cannot make
    # cpu-4 accidentally resolve an inline helper from cpu-2's -O0 binary.
    # pybind11 explicitly exports each module's Python initialization symbol.
    flags = ["-g0", "-fvisibility=hidden", "-ffp-contract=off"]
    if variant < 4:
        flags += ["-O0", "-fno-tree-vectorize", "-fno-tree-slp-vectorize"]
    else:
        # Tune scheduling for the Pi 5's Cortex-A76 without enabling fast-math
        # or increasing the baseline AArch64 instruction-set requirement.
        flags += ["-O3", "-mtune=cortex-a76"]

    link_flags = []
    parallel_backend = "serial"
    if threaded:
        parallel_backend = "OpenMP" if uses_openmp else "native"
        if uses_openmp:
            # ATen's header-only parallel_for otherwise compiles to a serial
            # fallback, even when the installed PyTorch supports OpenMP.
            flags.append("-fopenmp")
            link_flags.append("-fopenmp")

    name = "codegemm_cpu_kernel_1_mt" if scalar_mt else (
        "codegemm_cpu_kernel" if variant == 1 else f"codegemm_cpu_kernel_{variant}"
    )
    source = "kernels_1_mt.cpp" if scalar_mt else (
        "kernels.cpp" if variant == 1 else f"kernels_{variant}.cpp"
    )
    extensions.append(
        CppExtension(
            name=name,
            # Give each binary its own object filename. This also remains safe
            # if setuptools builds several extensions concurrently.
            sources=[source],
            # Distutils, as well as Ninja, must notice header/flag changes.
            depends=["setup.py", "build_info.h"] + (
                ["neon_bindings.h", "neon_math.h"] if variant > 1 else []
            ) + (["cpu_parallel.h"] if scalar_mt or variant > 1 else []),
            define_macros=[
                ("CODEGEMM_CPU_VARIANT", str(variant)),
                ("CODEGEMM_CPU_FLAGS", '"' + " ".join(flags) + '"'),
                ("CODEGEMM_CPU_PARALLEL_BACKEND", '"' + parallel_backend + '"'),
            ],
            extra_compile_args=flags,
            extra_link_args=link_flags,
        )
    )

setup(
    name="codegemm_cpu_kernel",
    version="0.3.0",
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
