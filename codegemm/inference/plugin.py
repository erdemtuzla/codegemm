# Originally from https://github.com/snu-mllab/GuidedQuant/blob/main/inference/plugin.py

import torch

_CUDA_EXTENSION = None

def _cuda_extension():
    global _CUDA_EXTENSION

    if _CUDA_EXTENSION is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The CUDA backend was selected, but CUDA is unavailable. "
                "Use backend='cpu' for CPU inference"
            )

        try:
            import codegemm_kernel
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "The CUDA backend requires the codegemm_kernel extension. "
                "Install the existing CUDA extension before using this backend."
            ) from exc

        _CUDA_EXTENSION = codegemm_kernel

    return _CUDA_EXTENSION
 

@torch.library.custom_op("plugin::codegemm_gemv", mutates_args={"output"})
def codegemm_gemv(
    x: torch.Tensor, 
    output: torch.Tensor, 
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    codebook: torch.Tensor, 
    group_size: int) -> None:
    _cuda_extension().codegemm_gemv(
        x, output, q_weight, alpha, codebook, group_size)

def codegemm_dequant(
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    codebook: torch.Tensor, 
    group_size: int) -> None:
    weight = _cuda_extension().codegemm_dequant(
        q_weight, alpha, codebook, group_size)
    return weight

@codegemm_gemv.register_fake
def _(x, output, q_weight, alpha, codebook, group_size):
    return None