import argparse
from datetime import datetime
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM
from codegemm.utils.run_logging import setup_run_log


def get_torch_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def read_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read().strip()
    return args.prompt


def build_inputs(tokenizer, prompt, args):
    if args.chat:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if tokenizer.chat_template is None:
            raise ValueError("This tokenizer does not define a chat template. Rerun without --chat.")
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    return tokenizer(prompt, return_tensors="pt").input_ids


def infer_model_kind(model_path):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        if "codegemm" in config or config.get("quantization_config", {}).get("quant_method") == "aqlm":
            return "codegemm"
    return "hf"


def load_model(args):
    model_kind = infer_model_kind(args.model_path) if args.model_kind == "auto" else args.model_kind
    if model_kind == "codegemm":
        return CodeGEMMForCausalLM.from_quantized(args.model_path).to(args.device).eval(), model_kind

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=get_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    return model.to(args.device).eval(), model_kind


def append_generation_result(output_file, model_path, record):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as results_file:
            results = json.load(results_file)
    else:
        results = {}

    results.setdefault(model_path, []).append(record)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a packed CodeGEMM or Hugging Face model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="packed_hf/tinyllama-codegemm-better",
        help="Packed CodeGEMM model path. Default: packed_hf/tinyllama-codegemm-better",
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="auto",
        choices=["auto", "codegemm", "hf"],
        help="Model loader to use. auto detects local CodeGEMM configs, hf loads a normal Hugging Face model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path or Hugging Face repo. Default: model_path.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="dtype for Hugging Face models. Ignored for CodeGEMM models.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantization in large language models in simple terms.",
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Optional text file to read the prompt from. Overrides --prompt.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt used with --chat.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use the tokenizer chat template before generation.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        help="Minimum number of new tokens to generate. Useful for stable speed benchmarks.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling. Use --no-do_sample for greedy decoding.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="history",
        help="Directory for timestamped run logs. Default: history",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional JSON file to append generated text and benchmark results.",
    )
    args = parser.parse_args()

    log_path = setup_run_log("run_generate", args.log_dir, args.model_path)
    print(f">> Run log: {log_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt = read_prompt(args)
    if not prompt:
        raise ValueError("Prompt is empty.")

    print(f">> Model: {args.model_path}")
    print(f">> Device: {args.device}")
    print(f">> Prompt: {prompt}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, model_kind = load_model(args)
    print(f">> Loader: {model_kind}")

    input_ids = build_inputs(tokenizer, prompt, args).to(args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "num_return_sequences": args.num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    prompt_tokens = input_ids.shape[-1]
    generated_tokens = sum(max(output_ids.shape[-1] - prompt_tokens, 0) for output_ids in outputs)
    tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else float("inf")

    print("\n---------------------- Generated Text ----------------------")
    generated_texts = []
    for index, output_ids in enumerate(outputs, start=1):
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_texts.append(text)
        if args.num_return_sequences > 1:
            print(f"\n[{index}]")
        print(text)

    print("\n---------------------- Benchmark ----------------------")
    print(f">> prompt_tokens={prompt_tokens}")
    print(f">> generated_tokens={generated_tokens}")
    print(f">> generation_time_seconds={elapsed_seconds:.4f}")
    print(f">> tokens_per_second={tokens_per_second:.2f}")

    max_cuda_memory_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    if max_cuda_memory_allocated is not None:
        print(f"\n>> max_cuda_memory_allocated={max_cuda_memory_allocated:,}")

    if args.output_file:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_path": args.model_path,
            "model_kind": model_kind,
            "device": args.device,
            "prompt": prompt,
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "generation_time_seconds": elapsed_seconds,
            "tokens_per_second": tokens_per_second,
            "max_cuda_memory_allocated": max_cuda_memory_allocated,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_return_sequences": args.num_return_sequences,
            "generated_texts": generated_texts,
            "run_log": log_path,
        }
        append_generation_result(args.output_file, args.model_path, record)
        print(f">> Generation JSON appended to {args.output_file}")


if __name__ == "__main__":
    main()
