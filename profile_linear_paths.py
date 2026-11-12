#!/usr/bin/env python3
"""Measure how CodeGEMM layer time splits between reconstruction and Psumbook.

Every CodeGEMM layer call is timed and labelled by the path it takes:
  - psumbook:       the input holds one token (each decode step)
  - reconstruction: the input holds several tokens (the prompt, perplexity)
With --psumbook-only, multi-token inputs also use Psumbook, one token at a
time (the same replacement as run_benchmark_psumbook_only.py).

Each request mirrors run_benchmark.py's latency mode: one forward pass over
the whole prompt, then new_tokens - 1 single-token decode steps. Token values
do not affect timing, so a fixed random prompt is used. Layer times include
the Python wrapper in cpu_ops.py as well as the C++ kernel.
"""
import argparse
import csv
import statistics
import time
from collections import defaultdict

import torch

import codegemm.inference.cpu_ops as cpu_ops
from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM
from codegemm.inference.cpu_backends import resolve_cpu_threads
from run_benchmark_psumbook_only import psumbook_only_linear

PATHS = ("reconstruction", "psumbook")


class LayerTimer:
    def __init__(self, linear, psumbook_only):
        self.linear = linear
        self.psumbook_only = psumbook_only
        self.reset()

    def reset(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def __call__(self, x, *args, **kwargs):
        single_token = x.numel() == x.shape[-1]
        path = "psumbook" if single_token or self.psumbook_only else "reconstruction"
        start = time.perf_counter()
        output = self.linear(x, *args, **kwargs)
        self.seconds[path] += time.perf_counter() - start
        self.calls[path] += 1
        return output


@torch.inference_mode()
def request(model, timer, prompt, new_tokens):
    result = {}
    timer.reset()
    start = time.perf_counter()
    output = model(input_ids=prompt, use_cache=True)
    result["prefill_ms"] = (time.perf_counter() - start) * 1000
    for path in PATHS:
        result[f"prefill_{path}_ms"] = timer.seconds[path] * 1000
        result[f"prefill_{path}_calls"] = timer.calls[path]
    past, token = output.past_key_values, output.logits[:, -1:].argmax(-1)

    timer.reset()
    start = time.perf_counter()
    for _ in range(new_tokens - 1):
        output = model(input_ids=token, past_key_values=past, use_cache=True)
        past, token = output.past_key_values, output.logits[:, -1:].argmax(-1)
    result["decode_ms"] = (time.perf_counter() - start) * 1000
    for path in PATHS:
        result[f"decode_{path}_ms"] = timer.seconds[path] * 1000
        result[f"decode_{path}_calls"] = timer.calls[path]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="packed_hf/llama68m-codegemm-m2b8v8g128-cpucheck-v2")
    parser.add_argument("--backend", required=True,
                        choices=["cpu-1", "cpu-1-mt", "cpu-2", "cpu-3", "cpu-4"])
    parser.add_argument("--threads", type=int, default=None,
                        help="Auto: 1 for cpu-1/2, at most 4 for cpu-1-mt/3/4")
    parser.add_argument("--prompt-lengths", default="128,512")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--psumbook-only", action="store_true",
                        help="Disable reconstruction; multi-token inputs use Psumbook per token")
    parser.add_argument("--csv", default=None, help="Optional CSV file for the median results")
    args = parser.parse_args()

    threads = resolve_cpu_threads(args.backend, args.threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    linear = psumbook_only_linear if args.psumbook_only else cpu_ops.linear
    timer = LayerTimer(linear, args.psumbook_only)
    cpu_ops.linear = timer
    model = CodeGEMMForCausalLM.from_quantized(args.model_path, backend=args.backend).eval()
    mode = "psumbook-only" if args.psumbook_only else "current"

    rows = []
    for prompt_tokens in [int(value) for value in args.prompt_lengths.split(",")]:
        prompt = torch.randint(100, 30000, (1, prompt_tokens),
                               generator=torch.Generator().manual_seed(0))
        for _ in range(args.warmup):
            request(model, timer, prompt, args.new_tokens)
        results = [request(model, timer, prompt, args.new_tokens) for _ in range(args.repeats)]
        median = {key: statistics.median(r[key] for r in results) for key in results[0]}

        recon = median["prefill_reconstruction_ms"] + median["decode_reconstruction_ms"]
        psum = median["prefill_psumbook_ms"] + median["decode_psumbook_ms"]
        layers = recon + psum
        total = median["prefill_ms"] + median["decode_ms"]
        row = {
            "backend": args.backend, "threads": threads, "mode": mode,
            "prompt_tokens": prompt_tokens, "new_tokens": args.new_tokens,
            **median,
            "reconstruction_share_of_layer_time": recon / layers,
            "psumbook_share_of_layer_time": psum / layers,
            "layer_share_of_request_time": layers / total,
        }
        rows.append(row)

        print(f"\n{args.backend}, {threads} thread(s), {mode}, prompt {prompt_tokens}, "
              f"{args.new_tokens} new tokens (median of {args.repeats})")
        for phase in ("prefill", "decode"):
            print(f"  {phase:7}: total {median[f'{phase}_ms']:10.1f} ms | "
                  f"reconstruction {median[f'{phase}_reconstruction_ms']:10.1f} ms "
                  f"({median[f'{phase}_reconstruction_calls']:4.0f} calls) | "
                  f"psumbook {median[f'{phase}_psumbook_ms']:10.1f} ms "
                  f"({median[f'{phase}_psumbook_calls']:4.0f} calls)")
        print(f"  layer time: reconstruction {100 * recon / layers:5.1f}% | "
              f"psumbook {100 * psum / layers:5.1f}% | "
              f"layers are {100 * layers / total:5.1f}% of the whole request")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
