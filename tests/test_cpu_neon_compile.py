"""Check explicit ARM SIMD code generation without claiming native execution.

WSL can check the arithmetic header with an AArch64 cross-compiler. The Python
numerical tests still have to execute cpu-2/3/4 on ARM before validating a Pi
installation; an assembly check cannot replace those runtime tests.
"""

from pathlib import Path
import platform
import shutil
import subprocess

import pytest


PROBE = r"""
#include "neon_math.h"
using namespace codegemm_cpu_neon;
extern "C" void transpose_probe(const float* input, float* output) {
    transpose_codebooks(input, output);
}
extern "C" void psum_probe(const float* input, const float* books, float* output) {
    build_psum_task(input, books, output, 0);
}
extern "C" void lookup_probe(const int32_t* codes, const float* scales,
                            const float* psum, float* output) {
    lookup_rows(codes, scales, psum, Shape{256, 131, 8}, 0, output, 0, 131);
}
extern "C" void reconstruct_probe(const int32_t* codes, const float* scales,
                                 const float* books, float* output) {
    reconstruct_rows(codes, scales, books, Shape{256, 131, 8}, 0, output, 0, 131);
}
"""


@pytest.mark.parametrize("optimized", (False, True), ids=("explicit-neon-O0", "neon-O3"))
def test_neon_header_emits_vector_instructions(tmp_path, optimized):
    compiler = shutil.which("aarch64-linux-gnu-g++")
    if compiler is None and platform.machine().lower() in ("aarch64", "arm64"):
        compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("No native or cross AArch64 C++ compiler is installed")
    kernel_dir = Path(__file__).resolve().parents[1] / "codegemm/inference/cpu_kernel"
    source = tmp_path / "probe.cpp"
    assembly = tmp_path / "probe.s"
    source.write_text(PROBE)
    flags = ["-O3", "-mtune=cortex-a76"] if optimized else [
        "-O0", "-fno-tree-vectorize", "-fno-tree-slp-vectorize"
    ]
    result = subprocess.run(
        [compiler, "-std=c++17", "-ffp-contract=off", *flags,
         "-I", str(kernel_dir), "-S", str(source), "-o", str(assembly)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generated = assembly.read_text().lower()
    # The -O0 case is the key check: these SIMD instructions must come from
    # our intrinsics, rather than from automatic loop vectorization.
    assert "fmla" in generated
    assert ".4s" in generated
