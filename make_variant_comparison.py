#!/usr/bin/env python3
"""Build cpu_variant_comparison.csv from run_benchmark.py JSON reports.

Example:
    python make_variant_comparison.py benchmarks/rpi_latency_lmhead_fix_01
    python make_variant_comparison.py OUT --pattern "{variant}_half_wt.json"
"""
import argparse
import csv
import json
from pathlib import Path

VARIANTS = ["cpu-1", "cpu-1-mt", "cpu-2", "cpu-3", "cpu-4"]
COLUMNS = ["backend", "threads", "dtype", "prompt_tokens", "new_tokens", "median_ttft_ms",
           "median_decode_tokens_per_second", "median_prefill_tokens_per_second"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="{variant}_latency.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: FOLDER/cpu_variant_comparison.csv")
    parser.add_argument("--with-perplexity", action="store_true",
                        help="Append perplexity, scored chunks and linear-path mode columns")
    args = parser.parse_args()
    columns = COLUMNS + (["perplexity", "ppl_chunks", "linear_path_mode"]
                         if args.with_perplexity else [])

    rows = []
    for variant in VARIANTS:
        path = args.folder / args.pattern.format(variant=variant)
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        meta = report["metadata"]
        extra = {}
        if args.with_perplexity:
            # One perplexity dataset per report in these experiments.
            quality = next(iter(report.get("quality", {}).values()), {})
            extra = {
                "perplexity": quality.get("perplexity"),
                "ppl_chunks": quality.get("chunks_evaluated"),
                "linear_path_mode": meta.get("linear_path_mode", "default"),
            }
        for generation in report["generation"]:
            summary = generation["latency_summary"]
            rows.append({
                **extra,
                "backend": meta["backend"],
                "threads": meta["threads"],
                "dtype": meta["dtype"],
                "prompt_tokens": generation["prompt_tokens"],
                "new_tokens": generation["new_tokens"],
                "median_ttft_ms": summary["ttft_ms"]["median"],
                "median_decode_tokens_per_second": summary["decode_tokens_per_second"]["median"],
                "median_prefill_tokens_per_second": summary["prefill_tokens_per_second"]["median"],
            })
    rows.sort(key=lambda row: (row["prompt_tokens"], VARIANTS.index(row["backend"])))

    output = args.output or args.folder / "cpu_variant_comparison.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
