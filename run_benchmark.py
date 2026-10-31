#!/usr/bin/env python3
"""Local CodeGEMM benchmark: frozen-token quality, latency, and memory.

Run from the repository root, or copy this file there before running it.
No model/kernel source changes are required. Batch size is deliberately one.
Hardware performance counters are collected separately with Linux perf.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import stat
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil
import torch


MIB = 1024 ** 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args):
    """Metadata probes run outside inference timing and never use a shell."""
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def pi_status():
    if not shutil.which("vcgencmd"):
        return None
    return {
        name: command(["vcgencmd", name, *extra])
        for name, extra in (
            ("measure_temp", []),
            ("measure_clock", ["arm"]),
            ("get_throttled", []),
        )
    }


def synchronize(device):
    # CPU measurements must not query or initialize CUDA.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def process_snapshot():
    process = psutil.Process()
    cpu = process.cpu_times()
    switches = process.num_ctx_switches()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_seconds": cpu.user + cpu.system,
        "voluntary_switches": switches.voluntary,
        "involuntary_switches": switches.involuntary,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
    }


def process_delta(before, elapsed):
    after = process_snapshot()
    delta = {key: after[key] - before[key] for key in before}
    # 100% means one fully occupied logical CPU; multithreading can exceed 100%.
    delta["cpu_utilization_percent_one_core_100"] = (
        100 * delta["cpu_seconds"] / elapsed
    )
    return delta


def describe(values):
    result = {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": None,
    }
    if len(values) >= 30:
        ordered = sorted(values)
        result["p95"] = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return result


def export_csv(path, report):
    """Compact scalar summaries for spreadsheets; raw repetitions stay in JSON."""
    rows = []
    meta = report["metadata"]
    common = [meta["backend"], meta["threads"], meta["dtype"]]
    def add(case, metric, value, unit):
        if value is not None:
            rows.append([case, metric, value, unit, *common])
    add("loading", "model_load_seconds", report["loading"]["seconds"], "s")
    add("loading", "rss_after_load_mib", report["loading"]["rss_after_load_mib"], "MiB")
    for item in report.get("generation", []):
        case = f"prompt{item['prompt_tokens']}_new{item['new_tokens']}"
        for metric, summary in item.get("latency_summary", {}).items():
            unit = "tokens/s" if metric.endswith("per_second") else "ms/token" if metric == "decode_ms_per_token" else "ms"
            add(case, "median_" + metric, summary["median"], unit)
        for metric, value in item.get("memory", {}).items():
            if metric.endswith("_mib"):
                add(case, metric, value, "MiB")
    for name, result in report.get("quality", {}).items():
        add(name, "perplexity", result["perplexity"], "ratio")
        add(name, "next_token_predictions_scored", result["next_token_predictions_scored"], "tokens")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "metric", "value", "unit", "backend", "threads", "dtype"])
        writer.writerows(rows)
    temporary.replace(path)


class PerfControl:
    """Handshake with external perf stat --control=fifo:CONTROL,ACK.

    perf starts disabled (-D -1). Only measured requests enable counters;
    model loading and warmup are excluded. Use a separate profiling run:
    control handshakes perturb wall-clock latency.
    """

    def __init__(self, control, acknowledgment, phase):
        for path in (control, acknowledgment):
            if not stat.S_ISFIFO(os.stat(path).st_mode):
                raise ValueError(f"Expected a named pipe: {path}")
        self.control = os.open(control, os.O_RDWR)
        self.ack = os.fdopen(os.open(acknowledgment, os.O_RDWR), "rb", buffering=0)
        self.phase = phase

    def send(self, message):
        os.write(self.control, (message + "\n").encode())
        # Some perf builds include a terminating NUL after each ack line.
        response = self.ack.readline().strip(b"\x00 \t\r\n")
        if response != b"ack":
            raise RuntimeError(f"Unexpected acknowledgment from perf stat: {response!r}")

    def begin(self, phase):
        if self.phase in (phase, "request") and phase == "prefill":
            self.send("enable")
        elif self.phase == phase == "decode":
            self.send("enable")

    def end(self, phase):
        if self.phase == phase or (self.phase == "request" and phase == "decode"):
            self.send("disable")


class MemoryProbe:
    """Sample RSS only in a separate memory pass, never the latency pass."""

    def __init__(self, device, interval):
        self.device = device
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.samples = []

    def sample(self):
        self.samples.append(self.process.memory_info().rss)

    def poll(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        synchronize(self.device)
        self.baseline = self.process.memory_info().rss
        self.sample()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        synchronize(self.device)
        self.stop_event.set()
        self.thread.join()
        self.sample()
        # On Linux ru_maxrss is KiB and is a lifetime high-water mark.
        self.result = {
            "rss_before_mib": self.baseline / MIB,
            "rss_sampled_peak_mib": max(self.samples) / MIB,
            "rss_sampled_growth_mib": (max(self.samples) - self.baseline) / MIB,
            "sampling_interval_ms": self.interval * 1000,
            "rss_samples": len(self.samples),
            "process_lifetime_peak_rss_mib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            ),
            "cuda_peak_allocated_mib": None,
            "cuda_peak_reserved_mib": None,
        }
        if self.device.type == "cuda":
            self.result.update({
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(self.device) / MIB
                ),
                "cuda_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(self.device) / MIB
                ),
            })


def load_tokens(path):
    # The transferred caches created by this repo contain plain tensor dicts.
    # Explicit weights_only=True avoids loading arbitrary serialized objects.
    data = torch.load(path, map_location="cpu", weights_only=True)
    ids = data["input_ids"]
    if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"Expected int64 input_ids with shape [1, tokens]: {path}")
    return ids.contiguous()


@torch.inference_mode()
def generation_request(model, prompt, generated_tokens, device, counters=None):
    """Model-only wall time: inputs/masks are prepared before the timer."""
    prompt_length = prompt.shape[1]
    mask = torch.ones(
        (1, prompt_length + generated_tokens), dtype=torch.long, device=device
    )
    token_history = []
    synchronize(device)
    if counters:
        counters.begin("prefill")
    start = time.perf_counter()

    output = model(
        input_ids=prompt,
        attention_mask=mask[:, :prompt_length],
        use_cache=True,
        return_dict=True,
    )
    synchronize(device)
    prefill_end = time.perf_counter()
    if counters:
        counters.end("prefill")
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = output.past_key_values
    del output
    token_history.append(token)
    synchronize(device)
    first_token_end = time.perf_counter()
    if counters:
        counters.begin("decode")

    # The first generated token came from prefill, so only G-1 steps remain.
    # Ignore EOS deliberately: every request performs the same amount of work.
    for index in range(1, generated_tokens):
        output = model(
            input_ids=token,
            attention_mask=mask[:, :prompt_length + index],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = output.past_key_values
        token_history.append(token)
        del output

    synchronize(device)
    end = time.perf_counter()
    if counters:
        counters.end("decode")
    decode_seconds = end - first_token_end
    result = {
        "prefill_ms": (prefill_end - start) * 1000,
        "ttft_ms": (first_token_end - start) * 1000,
        "decode_total_ms": decode_seconds * 1000,
        "decode_ms_per_token": decode_seconds * 1000 / (generated_tokens - 1),
        "decode_tokens_per_second": (generated_tokens - 1) / decode_seconds,
        "prefill_tokens_per_second": prompt_length / (prefill_end - start),
        "generation_total_ms": (end - start) * 1000,
    }
    # Copy diagnostics after timing so device-to-host copies are excluded.
    generated_ids = torch.cat(token_history, dim=1).cpu().tolist()[0]
    return result, generated_ids


@torch.inference_mode()
def quality(model, ids, chunk_size, max_chunks, device):
    available = ids.shape[1] // chunk_size
    count = min(available, max_chunks) if max_chunks else available
    if count == 0:
        raise ValueError("The input cache contains no complete evaluation chunk")
    losses = []
    synchronize(device)
    start = time.perf_counter()
    for index in range(count):
        chunk = ids[:, index * chunk_size:(index + 1) * chunk_size].to(device)
        output = model(
            input_ids=chunk, labels=chunk, use_cache=False, return_dict=True
        )
        losses.append(output.loss.detach().float())
        del output, chunk
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"    quality chunks: {index + 1}/{count}", flush=True)
    mean_nll = torch.stack(losses).mean()
    perplexity = torch.exp(mean_nll).item()
    synchronize(device)
    elapsed = time.perf_counter() - start
    if not math.isfinite(perplexity):
        raise RuntimeError("Non-finite perplexity; inspect the checkpoint/backend")
    return {
        "perplexity": perplexity,
        "mean_nll": mean_nll.item(),
        "seconds": elapsed,
        "chunk_size": chunk_size,
        "chunks_evaluated": count,
        "available_complete_chunks": available,
        "all_complete_chunks_evaluated": count == available,
        "cache_tokens": ids.shape[1],
        "input_tokens_evaluated": count * chunk_size,
        "next_token_predictions_scored": count * (chunk_size - 1),
        "unused_cache_tokens": ids.shape[1] - count * chunk_size,
        "use_cache": False,
    }


def metadata(repo, checkpoint, device, args, cache_paths):
    packages = {}
    for name in ("torch", "transformers", "accelerate", "psutil", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    extension = importlib.util.find_spec(
        "codegemm_cpu_kernel" if device.type == "cpu" else "codegemm_kernel"
    )
    artifacts = sorted(
        path for path in checkpoint.rglob("*")
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model")
    )
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "packages": packages,
        "torch_build": torch.__version__,
        "torch_build_configuration": torch.__config__.show(),
        "git_commit": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_status": command(["git", "-C", str(repo), "status", "--short"]),
        "benchmark_sha256": sha256(__file__),
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            str(path.relative_to(checkpoint)): sha256(path) for path in artifacts
        },
        "token_cache_sha256": {name: sha256(path) for name, path in cache_paths.items()},
        "extension_sha256": sha256(extension.origin) if extension else None,
        "backend": args.backend,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "arguments": vars(args),
        "batch_size": 1,
        "timing_scope": "model-only wall time, includes Python and CUDA synchronization",
        "memory_scope": "separate post-warmup requests; sampled process RSS",
        "p95_policy": "nearest-rank, reported only with at least 30 repetitions",
        "hardware_counters": {"status": "not_collected", "tool": "optional external perf"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="CodeGEMM repository root")
    parser.add_argument("--model-path", required=True, help="Local packed checkpoint")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", help="Defaults to backend; CUDA may use cuda:0")
    parser.add_argument("--mode", choices=("quality", "latency", "memory", "all"), default="all")
    parser.add_argument("--datasets", default="wikitext2-llama68m-cpucheck-v1,c4-new-llama68m-cpucheck-v1")
    parser.add_argument("--tokenizer-type", default="llama", help="Prefix used in existing cache filenames")
    parser.add_argument("--token-cache-dir", help="Defaults to the repo's evaluation token cache")
    parser.add_argument("--prompt-lengths", default="128,512")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--ppl-max-chunks", type=int, default=0, help="0 evaluates all complete cached chunks; positive limits are smoke tests")
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--perf-control", help="Control FIFO from external perf stat")
    parser.add_argument("--perf-ack", help="Acknowledgment FIFO from external perf stat")
    parser.add_argument("--perf-phase", choices=("prefill", "decode", "request"), default="decode")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("Use positive repeats/threads and nonnegative warmup")
    if args.new_tokens < 2 or args.chunk_size < 2 or args.ppl_max_chunks < 0 or args.sample_ms <= 0:
        parser.error("Require new-tokens >= 2, chunk-size >= 2, nonnegative ppl-max-chunks and positive sample-ms")
    try:
        lengths = [int(value) for value in args.prompt_lengths.split(",")]
    except ValueError:
        parser.error("prompt-lengths must be comma-separated integers")
    if not lengths or min(lengths) < 2:
        parser.error("Use prompt lengths >= 2 to exercise the prefill matrix path")
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not datasets:
        parser.error("Provide at least one dataset cache name")
    device = torch.device(args.device or args.backend)
    if device.type != args.backend:
        parser.error("device type must match backend")
    if bool(args.perf_control) != bool(args.perf_ack):
        parser.error("Provide perf-control and perf-ack together")
    if args.perf_control and (args.mode != "latency" or args.backend != "cpu" or len(lengths) != 1):
        parser.error("perf control requires CPU latency mode and exactly one prompt length")
    output_path = Path(args.output).resolve()
    if output_path.suffix != ".json":
        parser.error("output must end in .json; a matching .csv summary is also written")
    csv_path = output_path.with_suffix(".csv")
    if output_path.exists() or csv_path.exists():
        parser.error(f"JSON/CSV output already exists; choose a new filename: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo = Path(args.repo).resolve()
    if not (repo / "codegemm").is_dir():
        parser.error("repo must contain the codegemm package directory")
    sys.path.insert(0, str(repo))
    from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM

    checkpoint = Path(args.model_path).resolve()
    if not (checkpoint / "config.json").is_file():
        parser.error("model-path must point to a local packed checkpoint directory")
    cache_dir = Path(args.token_cache_dir).resolve() if args.token_cache_dir else repo / "codegemm/evaluate/input_tokens_cache"
    names = datasets if args.mode in ("quality", "all") else datasets[:1]
    cache_paths = {
        name: cache_dir / f"dataloader-{args.tokenizer_type}-{name}-test.pt" for name in names
    }
    for path in cache_paths.values():
        if not path.is_file():
            parser.error(f"Missing frozen token cache (no download is attempted): {path}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(0)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was selected but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(0)

    counters = PerfControl(args.perf_control, args.perf_ack, args.perf_phase) if args.perf_control else None
    pi_before = pi_status()

    def save():
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output_path)
        export_csv(csv_path, report)

    print(f"Loading {checkpoint.name} on {device}...", flush=True)
    synchronize(device)
    start = time.perf_counter()
    model = CodeGEMMForCausalLM.from_quantized(str(checkpoint), backend=args.backend)
    model = model.to(device).eval()
    synchronize(device)
    loading = {
        "seconds": time.perf_counter() - start,
        "rss_after_load_mib": psutil.Process().memory_info().rss / MIB,
        "scope": "checkpoint/model construction and device placement; excludes Python imports and hashing; OS file cache uncontrolled",
    }
    # Hash checkpoint files after loading so hashing cannot prewarm its disk reads.
    print("Recording metadata and artifact hashes...", flush=True)
    report = {"metadata": metadata(repo, checkpoint, device, args, cache_paths), "loading": loading}
    report["metadata"]["pi_before"] = pi_before
    if counters:
        print("Profiling run: perf handshakes affect latency; use unprofiled runs for speed comparisons.", flush=True)
        report["metadata"]["hardware_counters"] = {
            "status": "external_perf",
            "phase": args.perf_phase,
            "scope": "selected phases across measured requests; includes Python and boundary-control overhead",
            "results": "written by perf stat to its separate output file",
        }
        report["metadata"]["timing_scope"] += "; profiling control overhead present"
    actual_config = model.model.config
    if actual_config.model_type != "llama":
        raise ValueError("This initial benchmark is validated for Llama checkpoints only")
    report["metadata"]["dtype"] = str(next(model.parameters()).dtype)
    report["metadata"]["attention_implementation"] = getattr(actual_config, "_attn_implementation", None)
    report["metadata"]["quantization"] = model.config.codegemm
    context_limit = actual_config.max_position_embeddings
    save()
    ids = load_tokens(cache_paths[datasets[0]])

    if args.mode in ("latency", "memory", "all"):
        for length in lengths:
            if length + args.new_tokens > context_limit or length > ids.shape[1]:
                raise ValueError(f"Prompt {length} plus output exceeds available context/tokens")
            prompt = ids[:, :length].to(device).contiguous()
            prompt_hash = hashlib.sha256(prompt.cpu().numpy().tobytes()).hexdigest()
            print(f"Prompt {length}: {args.warmup} warmups...", flush=True)
            for _ in range(args.warmup):
                generation_request(model, prompt, args.new_tokens, device)
            record = {"prompt_tokens": length, "new_tokens": args.new_tokens, "decode_steps": args.new_tokens - 1, "prompt_ids_sha256": prompt_hash}
            if args.mode in ("latency", "all"):
                rows = []
                before = process_snapshot()
                block_start = time.perf_counter()
                for repeat in range(args.repeats):
                    row, generated_ids = generation_request(model, prompt, args.new_tokens, device, counters)
                    rows.append(row)
                    if repeat == 0:
                        record["first_request_generated_ids"] = generated_ids
                block_seconds = time.perf_counter() - block_start
                record["process_activity_measured_request_block"] = process_delta(before, block_seconds)
                record["raw_latency"] = rows
                record["latency_summary"] = {key: describe([row[key] for row in rows]) for key in rows[0]}
                summary = record["latency_summary"]
                print(f"  median TTFT {summary['ttft_ms']['median']:.3f} ms; decode {summary['decode_tokens_per_second']['median']:.3f} tokens/s", flush=True)
            if args.mode in ("memory", "all"):
                print("  Separate memory measurement...", flush=True)
                with MemoryProbe(device, args.sample_ms / 1000) as probe:
                    generation_request(model, prompt, args.new_tokens, device)
                record["memory"] = probe.result
            report.setdefault("generation", []).append(record)
            del prompt
            save()

    if args.mode in ("quality", "all"):
        if args.chunk_size > context_limit:
            raise ValueError("Quality chunk size exceeds model context")
        report["quality"] = {}
        for name, path in cache_paths.items():
            print(f"Quality: {name}", flush=True)
            tokens = load_tokens(path)
            result = quality(model, tokens, args.chunk_size, args.ppl_max_chunks, device)
            report["quality"][name] = result
            print(f"  PPL {result['perplexity']:.6f}", flush=True)
            del tokens
            save()

    report["metadata"]["pi_after"] = pi_status()
    report["completed"] = True
    save()
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
