# CMPE590 Submission README

This repository is based on the CodeGEMM implementation. My work focused on making the project runnable and measurable in a course/submission setting: installing the kernel, producing packed CodeGEMM checkpoints, running generation and evaluation, and comparing different vector-length configurations.

## Changes Made By Me

- `cb3fab4` - `Add logging and update some scripts to run the project`
- `cbd9b64` - `Add support for different vector lengths`

## Installation And Environment Setup

Recommended base environment:

- Linux machine with an NVIDIA GPU
- CUDA toolkit with `nvcc`
- PyTorch with CUDA support
- `g++` and `ninja`
- Python environment compatible with the package dependencies in `setup.py`

The original project used this NVIDIA container:

```bash
nvcr.io/nvidia/pytorch:24.10-py3
```

Install the Python package from the repository root:

```bash
pip install -e .
```

If `ninja` is not already available:

```bash
pip install ninja
```

Build and install the custom CodeGEMM CUDA extension:

```bash
cd codegemm/inference/custom_kernel
source do_install.sh
cd ../../..
```

The current kernel `setup.py` compiles for H100 (`sm_90`) and A100 (`sm_80`). For other GPUs, edit `codegemm/inference/custom_kernel/setup.py` and enable the matching `-gencode` line before reinstalling the kernel.

Useful local directories:

```bash
mkdir -p hf_models quantized packed_hf history
```

For gated Hugging Face models such as Meta Llama, authenticate first:

```bash
huggingface-cli login
```

## Simple End-To-End Example

This example uses TinyLlama to keep the run smaller than the full Llama-3.1-8B workflow. It quantizes a tiny configuration, converts it to a packed Hugging Face-style CodeGEMM model, generates text, and evaluates Wikitext-2 perplexity.

```bash
BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
SAVE_NAME=tinyllama-ns16-ft0-cb2-b8-v8-seq512-smoke
CACHE_DIR=hf_models

CUDA_VISIBLE_DEVICES=0 python run_quant.py "$BASE_MODEL" wikitext2 \
  --cache_dir "$CACHE_DIR" \
  --nsamples 16 \
  --model_seqlen 512 \
  --val_size 0 \
  --num_codebooks 2 \
  --nbits_per_codebook 8 \
  --in_group_size 8 \
  --scale_group_size 128 \
  --lr 1e-4 \
  --finetune_max_epochs 0 \
  --local_batch_size 1 \
  --resume \
  --use_fast_tokenizer \
  --trust_remote_code \
  --save "quantized/$SAVE_NAME"

python run_convert.py \
  --model "$BASE_MODEL" \
  --in_path "quantized/$SAVE_NAME" \
  --out_path "packed_hf/$SAVE_NAME" \
  --save_safetensors \
  --load_model \
  --swap_in_place \
  --save_tokenizer \
  --cache_dir "$CACHE_DIR"

python run_generate.py \
  --model_path "packed_hf/$SAVE_NAME" \
  --prompt "Explain neural-network weight quantization in two short paragraphs." \
  --max_new_tokens 64 \
  --min_new_tokens 32 \
  --no-do_sample \
  --seed 1234 \
  --device cuda \
  --output_file generation_smoke.json

python run_eval.py \
  --model_path "packed_hf/$SAVE_NAME" \
  --datasets wikitext2 \
  --skip_lm_eval \
  --output_file results_smoke.json
```

Expected outputs:

- quantized layer files and `args.pt` under `quantized/$SAVE_NAME`
- packed model files under `packed_hf/$SAVE_NAME`
- timestamped logs under `history/`
- generation benchmark/result JSON in `generation_smoke.json`
- perplexity result JSON in `results_smoke.json`
