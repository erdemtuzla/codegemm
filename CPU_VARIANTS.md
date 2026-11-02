# CPU implementations and the scalar multithreaded control

These backends run the same packed checkpoint with different CPU kernels. They
do not change quantization, retrain weights, or change checkpoint tensor layouts.
All variants currently support `m=2, b=8, v=8, g=128`, with a positive input width
divisible by 128 and a positive output width. CPU floating-point computation is
FP32. The existing CUDA implementation is retained.

| Backend | Custom computation | Custom-kernel compiler optimization | Custom-kernel threads | Where it runs |
| --- | --- | --- | --- | --- |
| `cpu-1` | Scalar C++ | Disabled with `-O0`; automatic vectorization disabled | One | x86-64 WSL and 64-bit ARM |
| `cpu-1-mt` | Scalar C++ with parallel work | Disabled with `-O0`; automatic vectorization disabled | Up to four | x86-64 WSL and 64-bit ARM |
| `cpu-2` | Explicit ARM NEON | Disabled with `-O0`; automatic vectorization disabled | One | AArch64 ARM |
| `cpu-3` | The NEON implementation with parallel work | Disabled with `-O0`; automatic vectorization disabled | Up to four | AArch64 ARM |
| `cpu-4` | The same threaded NEON implementation | Enabled with `-O3` | Up to four | AArch64 ARM |

The old name `cpu` is an alias for `cpu-1`. The names select implementations;
PyTorch device placement is still `cpu`. For example, `.to("cpu-3")` is invalid:
use the backend argument to select the kernel and `.to("cpu")` for placement.

“Compiler optimization disabled” refers to the new custom extension. Installed
PyTorch, its matrix-multiplication library, attention, embeddings, and other
framework operations are already compiled independently and remain optimized.
The experiment does not rebuild PyTorch. In particular, prompt processing still
uses PyTorch matrix multiplication after reconstructing temporary weight blocks.

## How the implementation changes

For one input vector, all backends use CodeGEMM's Psumbook algorithm. They compute
input/centroid dot products, decode the existing packed indices, look up the
selected dot products, apply the appropriate row/group scales, and add bias.
They do not expand the entire layer into dense weights for this path.

`cpu-2` computes several centroid dot products together using NEON instructions.
It prepares a temporary arrangement of the codebooks that lets neighboring
SIMD lanes process different centroids. This preparation happens during the
call; the saved codebooks and their learned values are unchanged. NEON also
accelerates reconstruction of the eight-element centroid vectors used during
prompt processing.

`cpu-1-mt`, `cpu-3`, and `cpu-4` share work through the OpenMP runtime used by PyTorch, or
through PyTorch's native thread pool when that is the installed parallel backend.
They split independent Psumbook entries and output rows across workers. Each
output row has a single owner while it is being computed, avoiding races and
floating-point atomic additions. The implementation limits active custom-kernel
work to four workers, matching the Raspberry Pi 5's four CPU cores. Setting
PyTorch's thread count to one still gives a valid single-thread run of these
backends.

`cpu-4` compiles the threaded NEON source with optimization enabled. It does not
enable `-ffast-math`. Explicit NEON can still change rounding through its
floating-point operations, and changing library thread counts can change
accumulation order. Compare perplexity with an appropriate numerical tolerance;
bit-for-bit equality between all backends is not a requirement.

The implementation uses bounded scratch space and reconstructs at most 128
output rows at a time for multiple-vector inputs. It does not keep a dense FP32
copy of every compressed layer.

## Compare scalar multithreading with NEON multithreading

`cpu-1-mt` is an additional control, not a replacement for `cpu-1`. It retains
the scalar codebook dot products, packed-index decoding, scale boundaries and
per-row accumulation order of `cpu-1`. It parallelizes the same two stages as
`cpu-3`: 32 independent Psumbook tables followed by independent output rows.
Prompt weight reconstruction is also split by output row. Both implementations
call the same scheduling helper in `cpu_parallel.h`, which supplies the same
barriers, nested-region handling and four-worker limit.

Compare `cpu-1-mt --threads 4` with `cpu-3 --threads 4` on the same Pi. Both use
`-O0`, disable automatic vectorization and FMA contraction, and use four threads
for eligible framework operations as well as the custom kernels. Comparing to
`cpu-4` instead would also introduce compiler optimization into the experiment.

This comparison measures the NEON implementation as a whole, including its
temporary codebook transpose, SIMD lookup strategy and explicit fused arithmetic.
It is not a single-instruction microbenchmark. `cpu-1-mt` keeps `cpu-1`'s original
codebook layout and does not add the transpose or NEON FMA. The underlying
Psumbook algorithm and checkpoint remain the same. PyTorch itself remains an
optimized library and may use SIMD in every backend, including scalar ones.

For a quick generation-only comparison, after defining `PACKED_PATH` and
`WT_CASE`, run this on Pi:

```bash
for backend in cpu-1-mt cpu-3; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    python run_benchmark.py \
        --model-path "$PACKED_PATH" \
        --backend "$backend" --threads 4 \
        --mode latency --datasets "$WT_CASE" \
        --prompt-lengths 128,512 --new-tokens 64 \
        --warmup 2 --repeats 3 \
        --output "benchmarks/${backend}_neon_comparison.json" \
        || break
done
```

Use distinct output filenames when repeating. The same `--backend cpu-1-mt`
works with `profile_cpu.sh`, `run_eval.py`, `run_generate.py`, and the Python API.

## Build the extensions

Activate the existing CodeGEMM conda environment and start in the repository root:

```bash
conda activate codegemm
```

On an x86-64 WSL machine, the default build produces `cpu-1` and `cpu-1-mt`.
On a 64-bit ARM machine, the default build produces all five variants:

```bash
(
    cd codegemm/inference/cpu_kernel
    MAX_JOBS=1 python -m pip install --force-reinstall --no-build-isolation --no-deps .
)
```

`MAX_JOBS=1` limits compiler jobs to reduce build-time RAM use. It does not limit
inference to one CPU thread. `--no-deps` uses the already installed PyTorch
environment rather than downloading another one. `--force-reinstall` replaces
the installed extension package, including the former optimized scalar binary.
Changing `setup.py` alone does not change a binary already installed in conda.

To select the variants included in a build, set `CODEGEMM_CPU_VARIANTS`:

```bash
(
    cd codegemm/inference/cpu_kernel
    CODEGEMM_CPU_VARIANTS=cpu-1,cpu-1-mt,cpu-2,cpu-3,cpu-4 MAX_JOBS=1 \
      python -m pip install --force-reinstall --no-build-isolation --no-deps .
)
```

That example requires AArch64. Selecting `cpu-2`, `cpu-3`, or `cpu-4` on x86-64
fails clearly; it does not silently replace NEON with scalar code. Building a
selected subset installs that subset of the extension package. Use the default
ARM build when you want all five variants available together. Installing only
`cpu-1-mt` with `--force-reinstall` replaces the extension distribution with that
subset; include all five names to retain the original modules alongside it.

Check the native modules on the Pi after installing:

```bash
python - <<'PY'
import importlib
import torch  # Load PyTorch's shared libraries before importing C++ extensions.

modules = {
    "cpu-1": "codegemm_cpu_kernel",
    "cpu-1-mt": "codegemm_cpu_kernel_1_mt",
    "cpu-2": "codegemm_cpu_kernel_2",
    "cpu-3": "codegemm_cpu_kernel_3",
    "cpu-4": "codegemm_cpu_kernel_4",
}
for backend, name in modules.items():
    extension = importlib.import_module(name)
    print(backend, extension.__file__, extension.build_info())
PY
```

On WSL, check `codegemm_cpu_kernel` and `codegemm_cpu_kernel_1_mt`. The `build_info()` output identifies
the backend, NEON support, threading support, and compiler optimization. A stale
extension without this information must be rebuilt before comparing variants.

## Select a backend

Use either the new short flag or the existing flag name in the evaluation and
generation entry points:

```bash
python run_eval.py --model_path "$PACKED_PATH" \
    --backend cpu-3 --cpu-threads 4 \
    --datasets "$WT_CASE,$C4_CASE" --skip_lm_eval \
    --output_file results_cpu3.json
```

`--codegemm-backend cpu-3` is equivalent to `--backend cpu-3`. CUDA remains the
default in those existing entry points. Their CPU thread defaults are one for
`cpu-1`/`cpu-2`, and up to four available CPUs for `cpu-1-mt`/`cpu-3`/`cpu-4`.

For Python API use:

```python
import torch
from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM

# Set the process-wide PyTorch thread count before starting inference.
# Model and layer constructors do not silently change this global setting.
torch.set_num_threads(4)
model = CodeGEMMForCausalLM.from_quantized(
    "packed_hf/llama68m-codegemm-m2b8v8g128-cpucheck-v2",
    backend="cpu-3",
).eval()
```

Each custom operation has at most four active work partitions even if the Python
caller sets a higher PyTorch thread count. With OpenMP, the requested team itself
is capped at four. A native-thread-pool build uses up to four active tasks in
PyTorch's existing pool. Other PyTorch operations can still use the caller's
thread count. The benchmark and execution CLIs cap requested counts for
`cpu-1-mt`/`cpu-3`/`cpu-4` so their matrix multiplication and custom kernels use a consistent
four-core budget.
For clean `cpu-1` and `cpu-2` comparisons, explicitly use one thread. Increasing
their PyTorch thread count accelerates eligible framework operations but does
not make the custom scalar/NEON loops multithreaded.

## Benchmark the cumulative variants on Pi

Use the same checkpoint and the same transferred input-token caches for every
run. Set these variables in the Pi shell rather than assuming values from WSL
were transferred with the files:

```bash
export RUN_NAME="llama68m-codegemm-m2b8v8g128-cpucheck-v2"
export PACKED_PATH="packed_hf/$RUN_NAME"
export WT_CASE="wikitext2-llama68m-cpucheck-v1"
export C4_CASE="c4-new-llama68m-cpucheck-v1"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p benchmarks
```

The `v1` suffix in the dataset names identifies the frozen input caches, not the
model being tested. Adjust `PACKED_PATH` if your checkpoint is still under
`transferred-files/`. The caches belong in
`codegemm/evaluate/input_tokens_cache/` under the repository root.

The following loop compares one-thread baselines with four-thread variants:

```bash
for backend in cpu-1 cpu-1-mt cpu-2 cpu-3 cpu-4; do
    case "$backend" in
        cpu-1|cpu-2) threads=1 ;;
        cpu-1-mt|cpu-3|cpu-4) threads=4 ;;
    esac

    OMP_NUM_THREADS="$threads" \
    MKL_NUM_THREADS="$threads" \
    OPENBLAS_NUM_THREADS="$threads" \
    python run_benchmark.py \
        --model-path "$PACKED_PATH" \
        --backend "$backend" \
        --mode all \
        --threads "$threads" \
        --datasets "$WT_CASE,$C4_CASE" \
        --prompt-lengths 128,512 \
        --new-tokens 64 \
        --warmup 2 \
        --repeats 10 \
        --chunk-size 2048 \
        --output "benchmarks/${RUN_NAME}_rpi5_${backend}_t${threads}.json"
done
```

`all` evaluates every complete cached perplexity chunk as well as generation and
memory workloads. The C4 cache is the previously prepared subset, not all of
C4. For quick performance experiments, choose `--mode latency`; this avoids
repeating perplexity. Existing output files are protected, so choose a new
filename for another run.

Additional useful controls are `cpu-3 --threads 1` and `cpu-4 --threads 1`.
They show how the threaded implementation behaves with one worker. The main
original four-variant experiment measures cumulative changes; the extra controls help isolate
threading from compiler optimization. Keep prompt length, output length, warmup,
repetition count, and input hashes fixed across comparisons.

Treat decode throughput as the primary measure of the custom Psumbook kernel.
Prefill also measures reconstruction and optimized library matrix multiplication,
so its speedup can differ substantially from decoding. Library thread counts are
part of the experiment and are recorded in the report.

The JSON/CSV reports retain the selected backend and native extension identity.
In the JSON, inspect these metadata fields:

| Field | Meaning |
| --- | --- |
| `backend` | Canonical runtime name, such as `cpu-3` |
| `cpu_backend_build` | The native binary's reported variant, NEON/threading support, optimization, and compile flags |
| `threads` | The process's PyTorch intra-op thread count |
| `custom_kernel_thread_limit` | The maximum custom-kernel worker count for this run |
| `extension_sha256` | Identity of the installed extension binary |
| `kernel_source_sha256` | Identities of the CPU source and build files |
| `token_cache_sha256` | Identity of the frozen input data |

Record the actual installed binary after each rebuild, not just the source
revision. Pi temperature, clock, and throttling should remain comparable between
runs; counter/profile runs should be separate from the ordinary latency runs.

## Validate before drawing performance conclusions

Run the CPU tests with the real checkpoint enabled:

```bash
CODEGEMM_CPU_CHECKPOINT="$PWD/$PACKED_PATH" \
python -m pytest -q \
    tests/test_cpu_backend.py \
    tests/test_cpu_variants.py \
    tests/test_cpu_neon_compile.py
```

Synthetic tests compare kernels with an independent reconstructed-weight
reference, including high packed indices and distinct scale groups. The
checkpoint test exercises prompt processing and cached generation. Native ARM
tests are required for the NEON variants. The suite skips NEON cases if their
modules are not installed, including on ARM. Check the skipped-test summary
(`pytest -ra` shows the reasons) and the five `build_info()` results before
treating a passing run as validation of all five variants. An x86-64 run
validates both scalar implementations and dispatch/rejection behavior.
Cross-compiling ARM code checks compilation and generated instructions; it does
not validate native runtime behavior or Pi performance.

Neither explicit NEON nor extra threads guarantees a speedup for every matrix
shape. At `-O0`, loop and intrinsic bookkeeping can be expensive. Small workloads
can also be dominated by parallel dispatch overhead. Measure all variants on the
same Pi and report the results; do not infer latency improvements from the
backend names alone.
