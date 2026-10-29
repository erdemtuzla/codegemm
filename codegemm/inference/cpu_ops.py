import importlib

import torch


_CPU_EXTENSION = None
_OUTPUT_ROWS_PER_BLOCK = 128


def validate_config(
    in_features,
    out_features,
    vec_len,
    group_size,
    num_codebooks,
    nbits_per_codebook,
):
    metadata = (
        num_codebooks,
        nbits_per_codebook,
        vec_len,
        group_size,
    )

    if metadata != (2, 8, 8, 128):
        raise NotImplementedError(
            "CPU inference initially supports only "
            "m=2, b=8, v=8, g=128."
        )

    if in_features <= 0 or in_features % 128 != 0:
        raise ValueError(
            "CPU input features must be positive and divisible by 128."
        )

    if out_features <= 0:
        raise ValueError("CPU output features must be positive.")


def _cpu_extension():
    global _CPU_EXTENSION

    if _CPU_EXTENSION is None:
        try:
            _CPU_EXTENSION = importlib.import_module(
                "codegemm_cpu_kernel"
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "The CPU backend requires codegemm_cpu_kernel. "
                "Build codegemm/inference/cpu_kernel first."
            ) from exc

    return _CPU_EXTENSION


@torch.no_grad()
def linear(x, codes, scales, codebooks, group_size, bias=None):
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise ValueError("CPU activations must be CPU float32 tensors.")

    if x.ndim < 1 or x.numel() == 0:
        raise ValueError("CPU input must contain at least one vector.")

    if codes.ndim != 3:
        raise ValueError("codes must have shape [2, K/32, M].")

    in_features = codes.shape[1] * 32
    out_features = codes.shape[2]

    if x.shape[-1] != in_features:
        raise ValueError(
            f"Expected {in_features} input features, "
            f"received {x.shape[-1]}."
        )

    if bias is not None:
        if (
            bias.device.type != "cpu"
            or bias.dtype != torch.float32
            or bias.shape != (out_features,)
        ):
            raise ValueError(
                "CPU bias must be a float32 tensor with shape [M]."
            )

    extension = _cpu_extension()

    # These calls do not copy already-contiguous tensors.
    codes = codes.contiguous()
    scales = scales.contiguous()
    codebooks = codebooks.contiguous()

    flat = x.reshape(-1, in_features).contiguous()

    if flat.shape[0] == 1:
        output = extension.gemv(
            flat[0],
            codes,
            scales,
            codebooks,
            group_size,
        )
    else:
        output = torch.empty(
            (flat.shape[0], out_features),
            dtype=torch.float32,
            device="cpu",
        )

        for begin in range(0, out_features, _OUTPUT_ROWS_PER_BLOCK):
            end = min(begin + _OUTPUT_ROWS_PER_BLOCK, out_features)

            weight = extension.dequantize_block(
                codes,
                scales,
                codebooks,
                group_size,
                begin,
                end,
            )

            output[:, begin:end].copy_(flat @ weight.t())
            del weight

    output = output.reshape(x.shape[:-1] + (out_features,))

    if bias is not None:
        output.add_(bias)

    return output