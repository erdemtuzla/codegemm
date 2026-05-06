import argparse
import os

import torch
from transformers import AutoTokenizer

from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM
from codegemm.utils.run_logging import setup_run_log


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


def main():
    parser = argparse.ArgumentParser(description="Generate text with a packed CodeGEMM model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="packed_hf/tinyllama-codegemm-better",
        help="Packed CodeGEMM model path. Default: packed_hf/tinyllama-codegemm-better",
    )
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CodeGEMMForCausalLM.from_quantized(args.model_path).to(args.device).eval()
    input_ids = build_inputs(tokenizer, prompt, args).to(args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
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

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    print("\n---------------------- Generated Text ----------------------")
    for index, output_ids in enumerate(outputs, start=1):
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        if args.num_return_sequences > 1:
            print(f"\n[{index}]")
        print(text)

    if torch.cuda.is_available():
        print(f"\n>> max_cuda_memory_allocated={torch.cuda.max_memory_allocated():,}")


if __name__ == "__main__":
    main()
