i# Quantization on this WSL machine

Inspected on 2026-10-22. These commands were reconstructed from this checkout,
`quantized/*/args.pt`, previous run logs, and shell history.

## Environment checked

| Component | Observed value |
| --- | --- |
| OS | Ubuntu 24.04.4, WSL2, kernel 6.18.54.1 |
| CPU | Ryzen 7 7700, 16 logical CPUs |
| WSL memory | Approximately 23 GiB RAM and 6 GiB swap |
| Free filesystem space | Approximately 511 GiB |
| Conda environment | `/home/erdem/miniconda3/envs/codegemm` |
| Python | 3.10.20 |
| PyTorch | 2.11.0, built with CUDA 13.0 |
| GPU | RTX 4060, 8 GiB VRAM |
| Transformers / datasets | 4.45.0 / 2.19.0, matching setup.py |
| Triton / AQLM | 3.6.0 / 1.1.7 |
| Conda CUDA compiler | nvcc 13.2.78 |

`pip check` passed. Quantization, inference, and `codegemm_kernel` imports passed.
Outside the agent sandbox, PyTorch detected the GPU and completed a CUDA tensor
addition. Inside the sandbox, GPU access was blocked; that did not indicate a
broken WSL installation. The shell initially had conda `base` active.

The local TinyLlama tokenizer/config and the repository's cached WikiText train,
WikiText test, and C4 validation datasets loaded successfully in offline mode.
No new quantization run or model inference was performed during this inspection.

## What each script does

- `run_quant.py`: calibrates and quantizes the original model, saves transformer
  blocks as `0.pth` through `21.pth` for TinyLlama, plus `args.pt` and
  `not_quantized_weights.pt`. Its `--finetune_*` options tune individual blocks
  during quantization.
- `run_convert.py --swap_in_place`: packs those blocks for CodeGEMM inference in
  `packed_hf/`.
- `run_eval.py`: evaluates the packed model; `--skip_lm_eval` selects perplexity
  without downstream benchmark tasks.
- `run_generate.py`: generates text from a packed model.
- `run_finetune.py`: separate, optional PV tuning using teacher/student models
  and distributed training. It is not required to repeat your earlier PTQ runs.

The README's main example uses an 8B model, four GPUs, and 8192-token sequences.
Your previous local runs used TinyLlama on one GPU with shorter sequences.

## 1. Activate the existing environment

Run these commands in your WSL terminal from this repository:

```bash
cd /home/erdem/master/thesis/external-repos/codegemm
source /home/erdem/miniconda3/etc/profile.d/conda.sh
conda activate codegemm

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE="$PWD/hf_models"
export HF_DATASETS_CACHE="$PWD/.hf_datasets"

python -c 'import sys, torch; print(sys.executable); print(torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
/usr/lib/wsl/lib/nvidia-smi
```

The absolute `nvidia-smi` path works even if `/usr/lib/wsl/lib` is absent from
`PATH`. There is already an installed CUDA extension; reinstalling dependencies
or rebuilding it is not part of this workflow.

For the already cached TinyLlama/WikiText/C4 workflow, optionally enable offline
mode before starting Python:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Unset these two variables when intentionally downloading a different model or
dataset. `--cache_dir hf_models` alone does not redirect every dataset/tokenizer
loader, so the cache environment variables above keep the locations consistent.

## 2. Repeat the previous quality configuration

This reproduces the saved parameters of
`tinyllama-ns256-ft3-cb2-b8-seq1024-quality` in a new output directory.

```bash
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
RUN_NAME="tinyllama-ns256-ft3-cb2-b8-seq1024-quality-$(date +%Y%m%d-%H%M%S)"
SAVE_PATH="quantized/$RUN_NAME"
PACKED_PATH="packed_hf/$RUN_NAME"

# The repo saves complete Python layer objects. This setting is for loading
# your own trusted checkpoints when resuming with this PyTorch version.
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python run_quant.py "$MODEL_PATH" wikitext2 \
  --cache_dir hf_models \
  --devices cuda:0 \
  --nsamples 256 \
  --model_seqlen 1024 \
  --val_size 32 \
  --num_codebooks 2 \
  --nbits_per_codebook 8 \
  --in_group_size 8 \
  --scale_group_size 128 \
  --init_max_iter 100 \
  --max_epochs 100 \
  --steps_per_epoch 50 \
  --lr 1e-4 \
  --relative_mse_tolerance 0.01 \
  --finetune_max_epochs 3 \
  --finetune_lr 1e-5 \
  --finetune_batch_size 1 \
  --finetune_keep_best \
  --local_batch_size 1 \
  --offload_activations \
  --seed 0 \
  --resume \
  --save "$SAVE_PATH"
```

The previous quality run logged 37,622.9 seconds of quantization, approximately
10.5 hours, before evaluation. This is historical timing, not a runtime guarantee.

`--nsamples 256 --val_size 32` means 224 calibration sequences used for training
and 32 held out for validation. `--offload_activations` moves activation storage
to system RAM. `--finetune_max_epochs 3` performs block tuning within PTQ.

To resume after interruption, reuse the same `SAVE_PATH` and all the same
parameters. Do not regenerate `RUN_NAME`. Saved blocks are reused by `--resume`;
changing quantization settings requires a new directory.

For a quick reproduction of the old v8 `fast2` experiment, use a new name and
replace the quality command's values with:

| Option | fast2 value |
| --- | --- |
| `--nsamples` | 16 |
| `--model_seqlen` | 512 |
| `--val_size` | 2 |
| `--init_max_iter` | 5 |
| `--max_epochs` | 3 |
| `--steps_per_epoch` | 2 |
| `--relative_mse_tolerance` | 0.10 |
| `--finetune_max_epochs` | 0 |

Also add `--skip_out_loss`. Keeping activation offload is an additional memory
precaution; your saved fast2 configuration had it disabled. This profile tests
the workflow but previously produced poor model quality.

## 3. Pack, evaluate, and generate

After quantization, in the same shell:

```bash
python run_convert.py \
  --model "$MODEL_PATH" \
  --in_path "$SAVE_PATH" \
  --out_path "$PACKED_PATH" \
  --cache_dir hf_models \
  --save_safetensors --load_model --swap_in_place --save_tokenizer

python run_eval.py \
  --model_path "$PACKED_PATH" \
  --datasets wikitext2 \
  --skip_lm_eval \
  --output_file "results_${RUN_NAME}.json"

python run_generate.py \
  --model_path "$PACKED_PATH" \
  --chat \
  --prompt "Explain model quantization in simple terms." \
  --max_new_tokens 128
```

Use `--swap_in_place`: its loader explicitly supports the complete layer objects
saved by `run_quant.py`. The alternative conversion branch uses a different
loading path and is not the workflow reconstructed here.

To evaluate the existing quality model immediately after environment activation:

```bash
python run_eval.py \
  --model_path packed_hf/tinyllama-ns256-ft3-cb2-b8-seq1024-quality \
  --datasets wikitext2 --skip_lm_eval \
  --output_file results_quality_recheck.json
```

## Previous results and pitfalls

Recorded packed-model WikiText2 perplexity, where lower is better:

| Saved run | Perplexity |
| --- | ---: |
| `ns256-ft3-...-quality` | 11.93 |
| `ns128-ft2-...-high` | 12.82 |
| `ns128-ft1-...-medium` | 147.30 |
| `ns128-ft2-...-balanced-isolated` | 265.37 |
| `ns16-ft0-...-v8-...-fast2` | 31,494.40 |

Sources: `results_quality.json`, `results_accuracy.json`, and
`results_vector-length.json`. These are historical results, not new measurements.
The evaluation script uses 2048-token chunks, while `run_quant.py` evaluates at
its configured sequence length; their perplexity values need not match.

- Model and dataset are **positional** arguments to `run_quant.py`; use
  `python run_quant.py MODEL wikitext2`, not `--model_path` or `--dataset`.
- `run_quant.py` always evaluates WikiText2 and C4 after saving quantized blocks.
  There is no skip-evaluation CLI option in this checkout. A late C4 download/cache
  error can occur after quantization has completed; inspect the log and saved
  blocks before repeating hours of quantization. The required C4 validation cache
  loaded successfully during this inspection.
- Older logs show PyTorch weights-only loading failures when resuming saved layer
  objects. The scoped `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting above addresses
  loads that omit an explicit `weights_only` argument; it does not override
  explicitly requested `weights_only=True`.
- `--nbits_per_codebook 8` means 256 entries per codebook. With two codebooks and
  vectors of length eight, code indices use roughly `2 * 8 / 8 = 2` bits per
  quantized weight, plus scales and codebook overhead. It does not mean an 8-bit
  model. Embeddings and other unquantized weights add further storage.
- Logs are automatically written under `history/`. Existing
  `scripts/convert_fast2_runs.sh` and `scripts/eval_fast2_runs.sh` batch-process
  three previously named fast2 models; they do not start quantization.
