#!/usr/bin/env python3
"""Run run_benchmark.py with reconstruction disabled: Psumbook for every input.

Normally a CodeGEMM layer uses Psumbook only for a single token (decode) and
reconstructs temporary FP32 weight blocks for several tokens (prompt
processing and perplexity). This wrapper replaces the multi-token path with
the same single-token Psumbook kernel, called once per token. Kernels,
checkpoints and run_benchmark.py are unchanged; all arguments pass through.

Example:
    OMP_NUM_THREADS=4 python run_benchmark_psumbook_only.py --backend cpu-4 \
        --model-path packed_hf/MODEL --mode latency --output OUT/cpu-4_latency.json
"""
import json
import runpy
import sys
from pathlib import Path

import torch

import codegemm.inference.cpu_ops as cpu_ops

_reconstructing_linear = cpu_ops.linear


@torch.no_grad()
def psumbook_only_linear(x, codes, scales, codebooks, group_size, bias=None, *, backend="cpu"):
    rows = x.numel() // x.shape[-1] if x.ndim else 0
    if rows <= 1:
        return _reconstructing_linear(
            x, codes, scales, codebooks, group_size, bias, backend=backend)

    flat = x.reshape(rows, x.shape[-1]).contiguous()
    # The first token uses the normal entry point, so every input check runs.
    first = _reconstructing_linear(
        flat[:1], codes, scales, codebooks, group_size, None, backend=backend)
    extension = cpu_ops._cpu_extension(backend)
    codes = codes.contiguous()
    scales = scales.contiguous()
    codebooks = codebooks.contiguous()

    output = torch.empty((rows, codes.shape[2]), dtype=torch.float32)
    output[0] = first[0]
    for row in range(1, rows):
        output[row] = extension.gemv(flat[row], codes, scales, codebooks, group_size)

    output = output.reshape(x.shape[:-1] + (codes.shape[2],))
    if bias is not None:
        output.add_(bias)
    return output


def _output_path(argv):
    for index, value in enumerate(argv):
        if value == "--output" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if value.startswith("--output="):
            return Path(value.split("=", 1)[1])
    return None


def main():
    cpu_ops.linear = psumbook_only_linear
    benchmark = Path(__file__).resolve().parent / "run_benchmark.py"
    sys.argv[0] = str(benchmark)
    try:
        runpy.run_path(str(benchmark), run_name="__main__")
    finally:
        # Label the report so it cannot be mistaken for a normal run.
        output = _output_path(sys.argv[1:])
        if output is not None and output.is_file():
            report = json.loads(output.read_text())
            report["metadata"]["linear_path_mode"] = "psumbook_only"
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
